export const EXPERIMENT_CONTROL_OPTIONS = Object.freeze({
  policyEngine: Object.freeze([
    ['claude-code','Claude Code · legacy policy MCP'],
    ['dt-arms-upstream','DT Arms upstream · native red-team loop'],
  ]),
  harnessProtocol: Object.freeze([
    ['v1','v1 · frozen six-tool control'],
    ['lazy-schema-v2','lazy-schema-v2 · seven tools'],
  ]),
  planningStrategy: Object.freeze([
    ['current','current'],
  ]),
  feedbackMode: Object.freeze([
    ['disabled','Disabled'],
    ['final','Final response'],
    ['final+deterministic','Final + deterministic'],
    ['final+deterministic+digestor','Final + deterministic + digestor'],
  ]),
  victimHarness: Object.freeze([
    ['openclaw','OpenClaw'],
    ['claudesdk','Claude SDK'],
    ['openaisdk','OpenAI SDK'],
    ['langchain','LangChain'],
    ['googleadk','Google ADK'],
    ['pocketflow','PocketFlow'],
    ['strands','Strands SDK'],
    ['hermes','Hermes'],
  ]),
  dtArmsVictimHarness: Object.freeze([
    ['openclaw','OpenClaw'],
    ['openaisdk','OpenAI SDK'],
    ['pocketflow','PocketFlow'],
    ['langchain','LangChain'],
    ['claudesdk','Claude SDK'],
    ['googleadk','Google ADK'],
  ]),
});

function scalarValue(raw){
  const value=String(raw??'').trim().replace(/\s+#.*$/,'').trim();
  if((value.startsWith('"')&&value.endsWith('"'))||(value.startsWith("'")&&value.endsWith("'")))return value.slice(1,-1);
  return value;
}

export function readYamlControl(yaml,section,key,fallback=''){
  const lines=String(yaml??'').split(/\r?\n/);let inside=false;
  for(const line of lines){
    if(new RegExp(`^${section}:\\s*(?:#.*)?$`).test(line)){inside=true;continue;}
    if(inside&&/^\S/.test(line)&&!/^\s*#/.test(line))break;
    if(!inside)continue;
    const match=line.match(new RegExp(`^\\s+${key}:\\s*(.*?)\\s*$`));
    if(match)return scalarValue(match[1])||fallback;
  }
  return fallback;
}

export function writeYamlControl(yaml,section,key,value){
  const source=String(yaml??'');const trailing=source.endsWith('\n');const lines=source.split(/\r?\n/);
  if(trailing)lines.pop();
  const rendered=`  ${key}: ${value}`;let sectionIndex=-1;let sectionEnd=lines.length;
  for(let index=0;index<lines.length;index+=1){
    if(new RegExp(`^${section}:\\s*(?:#.*)?$`).test(lines[index])){sectionIndex=index;break;}
  }
  if(sectionIndex<0){
    if(lines.length&&lines.at(-1).trim())lines.push('');
    lines.push(`${section}:`,rendered);
    return `${lines.join('\n')}\n`;
  }
  for(let index=sectionIndex+1;index<lines.length;index+=1){
    if(/^\S/.test(lines[index])&&!/^\s*#/.test(lines[index])){sectionEnd=index;break;}
    if(new RegExp(`^\\s+${key}:`).test(lines[index])){
      lines[index]=rendered;
      return lines.join('\n')+(trailing?'\n':'');
    }
  }
  lines.splice(sectionEnd,0,rendered);
  return lines.join('\n')+(trailing?'\n':'');
}
