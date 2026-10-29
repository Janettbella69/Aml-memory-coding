"""Insert a server-block file into an nginx.conf before the closing brace of its http { } block (the last `}` in the
file), keeping a .bak copy; a second run with the same block is a no-op. Check with `nginx -t` and reload afterwards.
Usage: python3 nginx_insert.py <nginx.conf> <block.conf>"""
import shutil
import sys

conf, block = sys.argv[1], sys.argv[2]
text = open(conf, encoding="utf-8").read()
piece = open(block, encoding="utf-8").read()
marker = piece.strip().splitlines()[0]
if marker in text:
    print("already present:", marker)
    sys.exit(0)
i = text.rstrip().rfind("}")
if i < 0:
    sys.exit("no closing brace found in " + conf)
shutil.copy(conf, conf + ".bak")
open(conf, "w", encoding="utf-8").write(text[:i] + piece.rstrip("\n") + "\n" + text[i:])
print("inserted before the final '}' of", conf, "(backup:", conf + ".bak)")
