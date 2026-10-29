"""Parse the Project Gutenberg text of a public-domain play (Bayard Veiller, The Thirteenth Chair, 1916; eBook #14095)
into ScriptMem's conversation schema: one session per act, entries of type dialogue (speaker, text) or narration
(speaker None, text). Used only as OUR OWN proxy corpus for script-style memory (short turns, stage directions,
no timestamps); the questions asked about it are written by us."""
import collections, json, re, sys
src, out = sys.argv[1], sys.argv[2]
t = open(src, encoding='utf-8', errors='replace').read().replace('\r\n', '\n')
body = t[t.find('*** START OF'):t.find('*** END OF')]
body = body[body.find('\nACT I\n'):]
paras = [re.sub(r'\s+', ' ', p).strip() for p in re.split(r'\n\s*\n', body) if p.strip()]
NAMES = ["MRS\\.? CROSBY", "MISS EASTWOOD", "MRS\\.? TRENT", "MISS ERSKINE", "MISS STANDISH", "WALES' VOICE", "CROSBY", "WILLIAM", "WALES",
         "TRENT", "STANDISH", "MASON", "POLLOCK", "DONOHUE", "DUNN", "DOOLAN", "HELEN", "ROSALIE", "BUTLER", "POLICEMAN", "ALL", "VOICE"]
SPEECH = re.compile(r"^(" + "|".join(NAMES) + r")\b\s*((?:\((?:[^()]|\([^()]*\))*\)\s*)*)[\.:]?\s*(.*)$")
def clean(s): return re.sub(r'\s+', ' ', s.replace('_', '')).strip()
def canon(name):
    name = name.replace("MRS ", "MRS. ").title()
    return {"Wales' Voice": "Wales' voice"}.get(name, name)
sessions, cur, last_speaker = [], None, None
for p in paras:
    m = re.match(r'^ACT (I{1,3})$', p)
    if m:
        if len(sessions) == 3: break            # property plots and the setting plan follow Act III
        cur = {"act": m.group(1), "entries": []}; sessions.append(cur); last_speaker = None; continue
    if cur is None or p.startswith('[Illustration') or p.upper().startswith('CURTAIN'): continue
    if clean(p).startswith('The CURTAIN falls'):
        break                                   # the acting edition's property and lighting plots follow
    m = SPEECH.match(p)
    if m and re.match(r"^(seated|standing|is |walking|--)", clean(m.group(3) or '')):
        cur["entries"].append({"type": "narration", "speaker": None, "text": clean(p)}); continue   # "Discovered:" staging lists
    if m and (m.group(3) or m.group(2)):
        speaker = canon(m.group(1)); direction = clean(m.group(2) or ''); text = clean(m.group(3))
        cur["entries"].append({"type": "dialogue", "speaker": speaker, "text": (direction + " " + text).strip()})
        last_speaker = speaker
    elif p.startswith('(') or p.startswith('_') or last_speaker is None:
        cur["entries"].append({"type": "narration", "speaker": None, "text": clean(p)})
    else:                                        # a speech continued after an interposed stage direction
        cur["entries"].append({"type": "dialogue", "speaker": last_speaker, "text": clean(p)})
n = 0
for si, s in enumerate(sessions, 1):
    for ei, e in enumerate(s["entries"], 1):
        e["dia_id"] = f"D{si}:{ei}"; n += 1
json.dump({"source": "Project Gutenberg eBook #14095 (public domain in the USA)", "sessions": sessions}, open(out, 'w', encoding='utf-8'), ensure_ascii=False, indent=0)
print("acts", len(sessions), "entries", n, "chars", sum(len(e['text']) for s in sessions for e in s['entries']))
for s in sessions:
    d = [e for e in s["entries"] if e["type"] == "dialogue"]
    print(f"  ACT {s['act']}: dialogue {len(d)}, narration {len(s['entries']) - len(d)}, chars {sum(len(e['text']) for e in s['entries'])}, median dialogue chars {sorted(len(e['text']) for e in d)[len(d)//2]}")
print("speakers:", collections.Counter(e["speaker"] for s in sessions for e in s["entries"] if e["type"] == "dialogue").most_common(30))
for si, s in enumerate(sessions, 1):
    with open(f"act{si}.txt", "w", encoding="utf-8") as f:
        for e in s["entries"]:
            f.write(f"{e['dia_id'].split(':')[1]} {e['speaker'] or '·'}: {e['text']}\n")
