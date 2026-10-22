"""Read-only audit of staged documentation, local links and credential leakage."""
import json
import os
from pathlib import PurePosixPath
import re
import subprocess


def git(*args):
    return subprocess.check_output(['git',*args])


def main():
    names=[p for p in git('ls-files','-z').decode().split('\0') if p]
    included=set(names)
    keys=[os.environ[k].encode() for k in ['SILICONFLOW_API_KEY','OPENROUTER_API_KEY','OPENAI_API_KEY','HF_TOKEN']
        if len(os.environ.get(k,''))>=8]
    errors=[]
    markdown=0
    size=0
    for name in names:
        data=git('show',':'+name)
        size+=len(data)
        path=PurePosixPath(name)
        if name.startswith('results/') and name!='results/.gitkeep':
            errors.append(f'Generated experiment artifact is staged: {name}')
        if name.endswith('.zh.md') or path.name=='.env' or path.name.startswith('.env.') and path.name!='.env.example':
            errors.append(f'Local note/credential file is staged: {name}')
        if any(key in data for key in keys) or re.search(rb'-----BEGIN [A-Z ]*PRIVATE KEY-----',data):
            errors.append(f'Credential material detected in {name}')
        if path.suffix!='.md':continue
        markdown+=1
        text=data.decode('utf-8')
        if re.search('[\u3400-\u9fff]',text):
            errors.append(f'Non-English CJK content in staged documentation: {name}')
        for raw in re.findall(r'\]\(([^)\n]+)\)',text):
            target=raw.strip().split(' "')[0].strip('<>')
            if target.startswith(('http:','https:','mailto:','#')):continue
            target=target.split('#')[0]
            if not target:continue
            parts=[]
            for part in (path.parent/PurePosixPath(target)).parts:
                if part=='..':
                    if not parts:break
                    parts.pop()
                elif part!='.':parts.append(part)
            else:
                if '/'.join(parts) in included:continue
            errors.append(f'Local documentation link is absent from the commit: {name} -> {target}')
    print(json.dumps({'staged_files':len(names),'staged_bytes':size,'markdown_files':markdown,
        'credential_values_available_for_audit':len(keys),'errors':errors},indent=2))
    if errors:raise SystemExit(1)


if __name__=='__main__':main()
