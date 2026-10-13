"""Inference-only checkpoint publication and metric return over SSH jump host."""
import argparse,json,os,shlex,sys,time
from pathlib import Path
import paramiko
from experiments.llm_task_sft.src.bridge import connect,execute
from experiments.llm_task_sft.src.files import atomic_json
from .contract import plateau_status

def publish(files,source,destination):
    temporary=destination+'.part'
    files.put(str(source),temporary)
    try:files.posix_rename(temporary,destination)
    except OSError:files.rename(temporary,destination)

def main():
    p=argparse.ArgumentParser();p.add_argument('--profile',type=Path,required=True)
    a=p.parse_args();profile=json.loads(a.profile.read_text());credentials=json.load(sys.stdin)
    root=Path(profile['run']);completed=set();root.mkdir(parents=True,exist_ok=True)
    while True:
        jump,target=connect(profile,credentials)
        try:
            files=paramiko.SFTPClient.from_transport(target.get_transport(),window_size=64*1024*1024,max_packet_size=1024*1024)
            for ready in sorted((root/'training').glob('B*/checkpoint-*/evaluation_ready.json'),key=lambda p:-json.loads(p.read_text())['epoch']):
                key=str(ready.relative_to(root));meta=json.loads(ready.read_text());dest=profile['target_run']+'/models/'+ready.parent.parent.name+'/'+ready.parent.name
                try:files.stat(dest+'/evaluation_ready.json');completed.add(key);continue
                except FileNotFoundError:pass
                execute(target,'mkdir -p '+shlex.quote(dest))
                # Publish marker last; never copy optimizer or FP32 resume weights.
                publish(files,ready.parent/'weights.pt',dest+'/weights.pt')
                publish(files,ready.parent/'state.json',dest+'/state.json')
                publish(files,ready,dest+'/evaluation_ready.json');completed.add(key)
                atomic_json(root/'bridge_status.json',dict(published=len(completed),last=meta,updated=time.time()))
            # Compact scores return to training lanes; raw answers remain on evaluator.
            for base in profile['baselines']:
                remote=profile['target_run']+'/results/'+base+'/curve.json'
                try:
                    with files.open(remote) as handle:curve=json.load(handle)
                except FileNotFoundError:continue
                atomic_json(root/'results'/base/'curve.json',curve)
                try:files.get(profile['target_run']+'/results/'+base+'/report.md',str(root/'results'/base/'report.md'))
                except FileNotFoundError:pass
                # Operational stop policy can be confirmed without changing the
                # optimizer/data contract already held by active training ranks.
                stop=json.loads(a.profile.read_text())['early_stop']
                config=json.loads(Path(profile['config']).read_text())
                flag=plateau_status(curve,stop,config['training']['save_steps'])
                if flag:
                    atomic_json(root/'results'/base/'plateau.json',flag)
                    if stop['mode']=='automatic':(root/'training'/base/'stop_after_checkpoint').touch()
                else:(root/'results'/base/'plateau.json').unlink(missing_ok=True)
            files.close()
            atomic_json(root/'bridge_status.json',dict(status='running',published=len(completed),updated=time.time()))
        finally:target.close();jump.close()
        time.sleep(profile.get('interval_seconds',60))

if __name__=='__main__':main()
