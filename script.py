from pathlib import Path

dossier = Path(__file__).parent
print(dossier)
for p in dossier.iterdir():
    print(p)
    if p.is_dir():
        for f in p.iterdir():
            print(f)