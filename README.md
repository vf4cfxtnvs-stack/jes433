# Personalisierter Bibeltext mit Piper

Diese kleine Web-App nimmt einen Vornamen entgegen, erzeugt daraus deinen
personalisierten Text nach Jesaja 43,1-4 und laesst ihn mit Piper als echte
Audiodatei sprechen.

## Piper einrichten

Im Projektordner:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install piper-tts
python -m piper.download_voices de_DE-thorsten-medium --data-dir models
```

Danach liegen das Modell `de_DE-thorsten-medium.onnx` und die zugehoerige
`de_DE-thorsten-medium.onnx.json` unter `models/`.

## Starten

Danach reicht im Projektordner schon:

```bash
python3 app.py
```

Die App findet `Piper` automatisch auch in der lokalen `.venv`.

Danach ist die Seite unter `http://127.0.0.1:8000` erreichbar.

Wenn auf Port `8000` schon eine andere Version laeuft, kannst du alternativ zum
Beispiel so starten:

```bash
PORT=8010 python3 app.py
```

## Stimme anpassen

Langsamer sprechen:

```bash
PIPER_LENGTH_SCALE=1.2 PIPER_SENTENCE_SILENCE=0.3 python3 app.py
```

Hinweis: Bei `PIPER_LENGTH_SCALE` bedeutet `1.0` normal. Hoeher ist langsamer,
zum Beispiel `1.15`, `1.2` oder `1.3`.

Ein anderes Modell testen:

```bash
PIPER_MODEL_PATH=models/de_DE-thorsten_emotional-medium.onnx python3 app.py
```

## HTTP-Test

Seite aufrufen:

```bash
curl http://127.0.0.1:8000
```

Einen Namen per Formular-POST senden:

```bash
curl -X POST -d "first_name=Anna" http://127.0.0.1:8000
```

## QR-Code-Idee

Wenn du spaeter einen QR-Code erstellen willst, sollte dieser einfach auf die
Webadresse deiner veroeffentlichten Seite zeigen, zum Beispiel:

`https://deine-domain.de/`

Dann kann jede Person den QR-Code scannen, den Vornamen eingeben und sich den
personalisierten Text mit derselben Piper-Stimme vorlesen lassen.

## Deployment mit Domain

Empfehlung: Lass deine bestehende Webseite unangetastet und lege fuer diese App
eine eigene Subdomain an, zum Beispiel:

`https://segen.deine-domain.de/`

### Empfohlener Ablauf

1. Den Projektordner in ein Git-Repository legen und zu GitHub pushen.
2. Das Repository bei Render als Blueprint importieren.
3. Render liest die Datei `render.yaml` und erstellt daraus den Python-Webdienst.
4. In deinem Domain-Anbieter fuer `segen.deine-domain.de` den von Render
   angegebenen DNS-Eintrag setzen.
5. Danach zeigt dein QR-Code einfach auf `https://segen.deine-domain.de/`

### Wichtige Dateien dafuer

- `render.yaml` beschreibt den Webdienst
- `requirements.txt` installiert `piper-tts`
- `scripts/render-build.sh` laedt das Piper-Modell waehrend des Builds

### QR-Code danach erstellen

Sobald die finale URL feststeht, erzeugst du daraus einen QR-Code, der genau auf
die App zeigt, zum Beispiel auf:

`https://segen.deine-domain.de/`
