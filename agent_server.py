#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能体分享服务器 —— 纯 Python，无需安装任何第三方库
使用方法：修改下面的 my_agent() 函数，然后双击 start.bat 启动
"""

import json
from http.server import HTTPServer, BaseHTTPRequestHandler


# ╔══════════════════════════════════════════════════════════╗
# ║   ★ 第一步：在这里写你的智能体逻辑 ★                   ║
# ║                                                          ║
# ║   输入：用户发送的文字（字符串）                         ║
# ║   输出：你的智能体的回复（字符串）                       ║
# ╚══════════════════════════════════════════════════════════╝
import sys, os

# ── 把 src 目录加入 Python 路径 ──────────────────────────
_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from retriever import get_retriever

def my_agent(user_input: str) -> str:
    """知识库两级检索问答"""
    # ── 第一步：检索知识库 ──────────────────────────────
    try:
        r = get_retriever()
        out = r.search(user_input, top_k=3)
    except Exception as e:
        return (f"⚠️ 检索器启动失败：{e}\n"
                f"请先运行：python src/train_classify.py")

    hits = [h for h in out["results"] if h["score"] > 0.01]
    pred_cat = out["predicted_category"]

    if not hits:
        return "抱歉，知识库中未找到相关内容，请换个方式提问。"

    # ── 直接展示检索原文 ──────────────────────────
    cat_line = f"📂 预测类别：{pred_cat}\n\n" if pred_cat else ""
    lines = [cat_line + "📚 知识库检索结果：\n"]
    for i, h in enumerate(hits, 1):
        lines.append(f"{i}. 【{h['title']}】")
        lines.append(h["content"])
        if h.get("keywords"):
            lines.append(f"关键词：{h['keywords']}")
        lines.append("")
    return "\n".join(lines)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  以下是 Web 服务器代码，通常不需要修改
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_HTML = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>我的智能体</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:system-ui,-apple-system,sans-serif;background:#f0f2f5;height:100vh;display:flex;flex-direction:column;overflow:hidden}
header{background:#fff;border-bottom:1px solid #e8e8e8;padding:12px 20px;display:flex;align-items:center;gap:12px;flex-shrink:0}
.av{width:40px;height:40px;border-radius:12px;background:linear-gradient(135deg,#667eea,#764ba2);display:flex;align-items:center;justify-content:center;color:#fff;font-size:20px;flex-shrink:0}
header h1{font-size:16px;font-weight:600;color:#111}
header p{font-size:12px;color:#52c41a;margin-top:2px}
#chat{flex:1;overflow-y:auto;padding:18px 20px;display:flex;flex-direction:column;gap:14px}
.row{display:flex;align-items:flex-end;gap:8px}
.row.user{flex-direction:row-reverse}
.mini{width:28px;height:28px;border-radius:8px;flex-shrink:0;display:flex;align-items:center;justify-content:center;font-size:14px}
.mini.bot{background:linear-gradient(135deg,#667eea,#764ba2);color:#fff}
.mini.usr{background:#5b6cf6;color:#fff;font-size:12px}
.bub{max-width:70%;padding:10px 14px;border-radius:18px;font-size:14px;line-height:1.7;white-space:pre-wrap;word-break:break-word}
.user .bub{background:#5b6cf6;color:#fff;border-bottom-right-radius:4px}
.bot .bub{background:#fff;color:#222;border-bottom-left-radius:4px;box-shadow:0 1px 4px rgba(0,0,0,.08)}
.dots{background:#fff;border-radius:18px;border-bottom-left-radius:4px;box-shadow:0 1px 4px rgba(0,0,0,.08);padding:13px 16px;display:flex;gap:5px}
.dots span{width:7px;height:7px;border-radius:50%;background:#bbb;animation:dot 1s infinite}
.dots span:nth-child(2){animation-delay:.2s}
.dots span:nth-child(3){animation-delay:.4s}
@keyframes dot{0%,80%,100%{transform:scale(1);opacity:.5}40%{transform:scale(1.35);opacity:1}}
#bar{background:#fff;border-top:1px solid #e8e8e8;padding:12px 16px;display:flex;gap:10px;align-items:flex-end;flex-shrink:0}
#inp{flex:1;border:1.5px solid #e0e0e0;border-radius:12px;padding:9px 14px;font-size:14px;resize:none;outline:none;max-height:110px;line-height:1.6;font-family:inherit;transition:border-color .15s}
#inp:focus{border-color:#5b6cf6}
#btn{width:42px;height:42px;border:none;border-radius:12px;background:#5b6cf6;color:#fff;font-size:18px;cursor:pointer;display:flex;align-items:center;justify-content:center;flex-shrink:0;transition:background .15s}
#btn:hover{background:#4a5ae8}
#btn:disabled{background:#c5c9e0;cursor:not-allowed}
</style>
</head>
<body>
<header>
  <div class="av">&#129302;</div>
  <div>
    <h1>我的智能体</h1>
    <p>&#9679; 在线</p>
  </div>
</header>
<div id="chat">
  <div class="row bot">
    <div class="mini bot">&#129302;</div>
    <div class="bub">你好！有什么可以帮你的？&#128522;</div>
  </div>
</div>
<div id="bar">
  <textarea id="inp" rows="1" placeholder="输入消息，Enter 发送 / Shift+Enter 换行"
    oninput="autosize(this)" onkeydown="onkey(event)"></textarea>
  <button id="btn" onclick="send()">&#10148;</button>
</div>
<script>
const chat=document.getElementById('chat');
const inp=document.getElementById('inp');
const btn=document.getElementById('btn');

function autosize(el){
  el.style.height='auto';
  el.style.height=Math.min(el.scrollHeight,110)+'px';
}

function onkey(e){
  if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();send();}
}

function addBubble(text,who){
  const row=document.createElement('div');
  row.className='row '+who;
  const mini=document.createElement('div');
  mini.className='mini '+(who==='user'?'usr':'bot');
  mini.innerHTML=who==='user'?'&#128100;':'&#129302;';
  const bub=document.createElement('div');
  bub.className='bub';
  bub.textContent=text;
  if(who==='user'){row.append(bub,mini);}else{row.append(mini,bub);}
  chat.appendChild(row);
  chat.scrollTop=chat.scrollHeight;
}

function addTyping(){
  const row=document.createElement('div');
  row.className='row bot';
  const mini=document.createElement('div');
  mini.className='mini bot';
  mini.innerHTML='&#129302;';
  const dots=document.createElement('div');
  dots.className='dots';
  dots.innerHTML='<span></span><span></span><span></span>';
  row.append(mini,dots);
  chat.appendChild(row);
  chat.scrollTop=chat.scrollHeight;
  return row;
}

async function send(){
  const text=inp.value.trim();
  if(!text||btn.disabled)return;
  inp.value='';
  inp.style.height='auto';
  btn.disabled=true;
  addBubble(text,'user');
  const loader=addTyping();
  try{
    const res=await fetch('/api/chat',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({message:text})
    });
    const data=await res.json();
    loader.remove();
    addBubble(data.reply,'bot');
  }catch(err){
    loader.remove();
    addBubble('连接失败，请确认 agent_server.py 正在运行。','bot');
  }
  btn.disabled=false;
  inp.focus();
}
</script>
</body>
</html>"""


class _Handler(BaseHTTPRequestHandler):

    def do_GET(self):
        if self.path == '/':
            body = _HTML.encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_POST(self):
        if self.path == '/api/chat':
            try:
                length = int(self.headers.get('Content-Length', 0))
                raw = self.rfile.read(length).decode('utf-8')
                data = json.loads(raw)
                user_input = data.get('message', '').strip()
                reply = my_agent(user_input) if user_input else '请输入内容'
            except Exception as exc:
                reply = f'[服务器错误] {exc}'

            body = json.dumps({'reply': reply}, ensure_ascii=False).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, fmt, *args):
        # 只打印真正的请求，过滤健康检查等噪音
        if args and '200' in str(args[0]):
            print(f'  [{self.address_string()}] {args[0]}')


if __name__ == '__main__':
    PORT = 8000
    httpd = HTTPServer(('0.0.0.0', PORT), _Handler)
    print('=' * 50)
    print('  &#129302;  智能体服务器已启动')
    print(f'  本地预览：http://localhost:{PORT}')
    print('  等待 Cloudflare 隧道连接...')
    print('  按 Ctrl+C 停止')
    print('=' * 50)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print('\n已停止。')
        httpd.server_close()
