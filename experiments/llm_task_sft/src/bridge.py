"""Transfer complete HF epoch models via login02; credentials stay in memory."""

import argparse
import json
from pathlib import Path
import shlex
import sys
import tarfile
import time

import paramiko

from .files import atomic_json


def connect(profile, credentials):
    jump=paramiko.SSHClient(); jump.load_host_keys(profile['known_hosts']); jump.set_missing_host_key_policy(paramiko.RejectPolicy())
    jump.connect(profile['jump_host'],port=profile['jump_port'],username=profile['jump_user'],
                 password=credentials['jump_password'],look_for_keys=False,allow_agent=False,timeout=20)
    channel=jump.get_transport().open_channel('direct-tcpip',(profile['target_host'],22),('127.0.0.1',0))
    target=paramiko.SSHClient(); target.load_host_keys(profile['known_hosts']); target.set_missing_host_key_policy(paramiko.RejectPolicy())
    target.connect(profile['target_host'],username=profile['target_user'],password=credentials['target_password'],
                   look_for_keys=False,allow_agent=False,sock=channel,timeout=20)
    return jump,target


def execute(target,command):
    _,stdout,stderr=target.exec_command(command)
    output=stdout.read().decode(); error=stderr.read().decode(); code=stdout.channel.recv_exit_status()
    if code: raise RuntimeError(f'Remote artifact command failed ({code}): {error[:1000]}')
    return output


def scp(target, source, destination, profile, password):
    command=['scp','-o','StrictHostKeyChecking=yes','-o',f"UserKnownHostsFile={profile['target_known_hosts']}",
             '-P',str(profile['jump_port']),source,destination]
    stdin,stdout,_=target.exec_command(shlex.join(command),get_pty=True)
    channel=stdout.channel; recent=''; sent=False; deadline=time.monotonic()+1800
    while not channel.exit_status_ready():
        if time.monotonic()>deadline: channel.close(); raise TimeoutError('Artifact transfer exceeded 30 minutes')
        if channel.recv_ready():
            recent=(recent+channel.recv(32768).decode(errors='replace'))[-2000:]
            if 'password:' in recent.lower() and not sent:
                stdin.write(password+'\n'); stdin.flush(); sent=True; recent=''
        else: time.sleep(.1)
    code=channel.recv_exit_status()
    # Never include PTY contents in errors: a terminal could echo its password input.
    if code: raise RuntimeError(f'Artifact scp failed with return code {code}')


def model_archive_command(checkpoint,archive):
    code="""import pathlib,tarfile
p=pathlib.Path(CHECKPOINT)
with tarfile.open(ARCHIVE,'w') as stream:
 for item in p.iterdir():
  if item.is_file() and (item.suffix in ('.safetensors','.json','.txt','.model','.jinja')):
   stream.add(item,arcname=item.name)
""".replace('CHECKPOINT',repr(str(checkpoint))).replace('ARCHIVE',repr(str(archive)))
    return 'python3 -c '+shlex.quote(code)


def extract_model(archive,directory):
    directory.mkdir(parents=True,exist_ok=True)
    with tarfile.open(archive) as stream:
        members=stream.getmembers()
        if any(not m.isfile() or Path(m.name).name!=m.name for m in members):
            raise ValueError('Unexpected artifact archive member')
        for member in members:
            # Flat, regular, individually validated model files only.
            with stream.extractfile(member) as source, (directory/member.name).open('wb') as destination:
                import shutil
                shutil.copyfileobj(source,destination,1024*1024)
    if not (directory/'config.json').exists() or not list(directory.glob('*.safetensors')):
        raise ValueError('Model archive lacks complete HF weights')


def stage_data(profile,credentials):
    """One initial transfer; subsequent bridge work only sends epoch model weights."""
    data=Path(profile['data'])
    if not (data/'ready.json').exists(): raise ValueError('Training data is not ready')
    archive=Path(profile['run'])/'incoming'/'data_50k.tar'
    with tarfile.open(archive,'w') as stream:
        stream.add(data,arcname=data.name)
    jump,target=connect(profile,credentials)
    try:
        stage=profile['jump_stage']+'/data_50k.tar'
        files=paramiko.SFTPClient.from_transport(jump.get_transport(),window_size=64*1024*1024,max_packet_size=1024*1024)
        try: files.put(str(archive),stage)
        finally: files.close()
        remote_archive=profile['target_run']+'/incoming/data_50k.tar'
        scp(target,f"{profile['jump_user']}@{profile['jump_host']}:{stage}",remote_archive,profile,credentials['jump_password'])
        execute(target,shlex.join(['tar','xf',remote_archive,'-C',profile['target_run']]))
        atomic_json(Path(profile['run'])/'data_transfer_complete.json',dict(target_data=profile['target_run']+'/'+data.name))
    finally: target.close(); jump.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',type=Path,required=True)
    parser.add_argument('--stage-data',action='store_true')
    args=parser.parse_args(); profile=json.loads(args.profile.read_text())
    credentials=json.load(sys.stdin)  # supplied by a detached launcher through a pipe
    run=Path(profile['run']); (run/'incoming').mkdir(exist_ok=True); (run/'models').mkdir(exist_ok=True)
    if args.stage_data:
        stage_data(profile,credentials)
        return
    epochs=profile['epochs']; completed=[]
    while len(completed)<epochs:
        jump,target=connect(profile,credentials)
        try:
            with target.open_sftp() as files:
                try:
                    with files.open(profile['target_train']+'/progress.json') as source:
                        atomic_json(run/'training_progress.json',json.load(source))
                except FileNotFoundError: pass
                names=files.listdir(profile['target_train'])
                ready=[]
                for name in names:
                    if not name.startswith('checkpoint-'): continue
                    try:
                        with files.open(profile['target_train']+'/'+name+'/evaluation_ready.json') as source: item=json.load(source)
                        with files.open(profile['target_train']+'/'+name+'/training_state.json') as source: state=json.load(source)
                    except FileNotFoundError: continue
                    if state['smoke']: continue
                    ready.append((item['epoch'],name,item))
            for epoch,name,item in sorted(ready):
                destination=run/'models'/f'epoch_{epoch}'
                if (destination/'evaluation_ready.json').exists():
                    if epoch not in completed: completed.append(epoch)
                    continue
                started=time.monotonic(); archive_name=f'epoch_{epoch}.tar'
                target_archive=profile['target_run']+'/incoming/'+archive_name
                stage=profile['jump_stage']+'/'+archive_name
                execute(target,model_archive_command(profile['target_train']+'/'+name,target_archive))
                scp(target,target_archive,f"{profile['jump_user']}@{profile['jump_host']}:{stage}",profile,credentials['jump_password'])
                local_archive=run/'incoming'/archive_name
                files=paramiko.SFTPClient.from_transport(jump.get_transport(),window_size=64*1024*1024,max_packet_size=1024*1024)
                try: files.get(stage,str(local_archive),max_concurrent_prefetch_requests=32)
                finally: files.close()
                incoming=run/'incoming'/f'epoch_{epoch}'
                extract_model(local_archive,incoming)
                # The target marker in the archive is not published until all weights have arrived.
                (incoming/'evaluation_ready.json').unlink(missing_ok=True)
                incoming.replace(destination)
                atomic_json(destination/'evaluation_ready.json',dict(item,sync_seconds=time.monotonic()-started))
                completed.append(epoch)
                atomic_json(run/'bridge_status.json',dict(completed_epochs=completed,last_transfer_seconds=time.monotonic()-started,utc_epoch=time.time()))
        finally: target.close(); jump.close()
        if len(completed)<epochs: time.sleep(profile['interval_seconds'])
    atomic_json(run/'bridge_complete.json',dict(completed_epochs=completed))


if __name__=='__main__': main()
