#!/usr/bin/env python3
"""V4 storytelling video engine. Usage: python3 v4video.py <step> <work_dir>
Steps: setup | transcribe | faces | cropcheck | clips | audio | music | overlay | compose | qa | light
All project data lives in <work_dir>/config.json (see SKILL.md)."""
import glob, json, os, subprocess, sys

STEP, WD = sys.argv[1], os.path.abspath(sys.argv[2])
ASSETS = os.path.expanduser("~/v4assets")
FONTS = f"{ASSETS}/fonts"
CFG_PATH = f"{WD}/config.json"
cfg = json.load(open(CFG_PATH)) if os.path.exists(CFG_PATH) else {}
SRC = cfg.get("src", f"{WD}/ep.mp4")
FPS = 30
END = cfg.get("end_secs", 2.5)

def run(*a, **k):
    return subprocess.run(list(a), check=True, **k)

def probe():
    o = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                        "stream=width,height:format=duration", "-of", "json", SRC],
                       capture_output=True, text=True).stdout
    j = json.loads(o)
    return j["streams"][0]["width"], j["streams"][0]["height"], float(j["format"]["duration"])

# ---------------- timeline ----------------
def timeline():
    out, t = [], 0.0
    for i, (sp, a, b, z, act) in enumerate(cfg["segments"]):
        n = round((b - a) * FPS); d = n / FPS
        out.append(dict(i=i, sp=sp, a=a, b=a + d, z=z, act=act, n=n, o=t, d=d))
        t += d
    return out, t

def src2out(tl, ts):
    for s in tl:
        if s["a"] <= ts < s["b"]:
            return s["o"] + ts - s["a"]
    raise SystemExit(f"source time {ts} is not inside any segment")

# ---------------- steps ----------------
def setup():
    os.makedirs(FONTS, exist_ok=True)
    G = "https://raw.githubusercontent.com"
    files = {
        f"{FONTS}/Anton-Regular.ttf": f"{G}/google/fonts/main/ofl/anton/Anton-Regular.ttf",
        f"{FONTS}/BarlowCondensed-Bold.ttf": f"{G}/google/fonts/main/ofl/barlowcondensed/BarlowCondensed-Bold.ttf",
        f"{FONTS}/Montserrat-Black.ttf": f"{G}/JulietaUla/Montserrat/master/fonts/ttf/Montserrat-Black.ttf",
        f"{FONTS}/Montserrat-Bold.ttf": f"{G}/JulietaUla/Montserrat/master/fonts/ttf/Montserrat-Bold.ttf",
        f"{ASSETS}/sh.rnnn": f"{G}/richardpl/arnndn-models/master/sh.rnnn",
        f"{ASSETS}/front.xml": f"{G}/opencv/opencv/4.x/data/haarcascades/haarcascade_frontalface_default.xml",
        f"{ASSETS}/profile.xml": f"{G}/opencv/opencv/4.x/data/haarcascades/haarcascade_profileface.xml",
    }
    for dst, url in files.items():
        if not os.path.exists(dst):
            run("curl", "-sSfL", "-o", dst, url)
    try:
        import faster_whisper  # noqa
    except ImportError:
        run("pip", "install", "--break-system-packages", "-q", "faster-whisper")
    print("setup ok")

def transcribe():
    import wave, numpy as np
    from faster_whisper import WhisperModel
    run("ffmpeg", "-v", "error", "-y", "-i", SRC, "-vn", "-ac", "1", "-ar", "16000", f"{WD}/audio16k.wav")
    run("ffmpeg", "-v", "error", "-y", "-i", SRC, "-vn", "-c:a", "pcm_s16le", f"{WD}/audio_full.wav")
    w = wave.open(f"{WD}/audio16k.wav")
    a = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
    model = WhisperModel(cfg.get("whisper", "medium"), device="cpu", compute_type="int8")
    segs, _ = model.transcribe(a, language="pt", word_timestamps=True, vad_filter=True, beam_size=5)
    words, lines = [], []
    for s in segs:
        lines.append(f"[{s.start:7.2f}-{s.end:7.2f}] {s.text.strip()}")
        words += [{"s": x.start, "e": x.end, "w": x.word} for x in s.words]
    json.dump(words, open(f"{WD}/words.json", "w"), ensure_ascii=False)
    open(f"{WD}/transcript.txt", "w").write("\n".join(lines))
    print("\n".join(lines))

def faces():
    import cv2, numpy as np
    W, H, dur = probe()
    os.makedirs(f"{WD}/fd", exist_ok=True)
    step = max(6, dur / 40)
    t = 3.0
    while t < dur - 1:  # seek per frame: decoding the whole 4K file with an fps filter is far too slow
        run("ffmpeg", "-v", "error", "-y", "-ss", f"{t:.1f}", "-i", SRC, "-frames:v", "1",
            "-vf", "scale=1920:-2", f"{WD}/fd/{int(t):05d}.jpg")
        t += step
    k = W / 1920
    fr = cv2.CascadeClassifier(f"{ASSETS}/front.xml"); pr = cv2.CascadeClassifier(f"{ASSETS}/profile.xml")
    pts = []
    for f in sorted(glob.glob(f"{WD}/fd/*.jpg")):
        g = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2GRAY); gw = g.shape[1]
        fs = list(fr.detectMultiScale(g, 1.1, 6, minSize=(60, 60))) + list(pr.detectMultiScale(g, 1.1, 6, minSize=(60, 60)))
        fs += [(gw - x - w, y, w, h) for x, y, w, h in pr.detectMultiScale(cv2.flip(g, 1), 1.1, 6, minSize=(60, 60))]
        for x, y, w, h in fs:
            if h * k > H * 0.25 or (y + h / 2) * k > H * 0.75:
                continue
            pts.append(((x + w / 2) * k, (y + h / 2) * k, w * k))
    pts = sorted(pts)
    xs = [p[0] for p in pts]
    gaps = [(xs[i + 1] - xs[i], i) for i in range(len(xs) - 1)]
    clusters, start = [], 0
    cut = sorted(i for gsz, i in gaps if gsz > W * 0.12)
    for i in cut + [len(pts) - 1]:
        clusters.append(pts[start:i + 1]); start = i + 1
    print(f"source {W}x{H}, {len(glob.glob(WD + '/fd/*.jpg'))} frames sampled")
    for c in clusters:
        if len(c) < 4:
            continue
        a = np.array(c)
        print(f"face cluster: n={len(c)} cx={np.median(a[:,0]):.0f} cy={np.median(a[:,1]):.0f} size={np.median(a[:,2]):.0f}")

def crop_box(sp, z, W, H):
    s = cfg["speakers"][sp]
    h = round(H * (cfg.get("crop_h_punch", 0.49) if z == "p" else cfg.get("crop_h", 0.57)) / 2) * 2
    w = round(h * 9 / 16 / 2) * 2
    x = int(min(max(s["cx"] - w / 2, 0), W - w))
    y = int(min(max(s["cy"] - cfg.get("face_frac", 0.40) * h, 0), H - h))
    return w, h, x, y

def cropcheck():
    from PIL import Image
    W, H, _ = probe()
    f = sorted(glob.glob(f"{WD}/fd/*.jpg"))[len(glob.glob(f"{WD}/fd/*.jpg")) // 2]
    im = Image.open(f); k = im.size[0] / W
    tiles = []
    for sp in cfg["speakers"]:
        for z in "np":
            w, h, x, y = crop_box(sp, z, W, H)
            tiles.append(im.crop((int(x * k), int(y * k), int((x + w) * k), int((y + h) * k))).resize((360, 640)))
    g = Image.new("RGB", (360 * len(tiles), 640))
    for i, t in enumerate(tiles):
        g.paste(t, (i * 360, 0))
    g.save(f"{WD}/cropcheck.jpg", quality=80); print(f"{WD}/cropcheck.jpg")

def clips():
    W, H, _ = probe()
    tl, total = timeline()
    os.makedirs(f"{WD}/clips", exist_ok=True)
    for s in tl:
        f = f"{WD}/clips/c{s['i']:02d}.mp4"
        if os.path.exists(f):
            continue
        w, h, x, y = crop_box(s["sp"], s["z"], W, H)
        vf = f"crop={w}:{h}:{x}:{y},scale=1080:1920:flags=lanczos,fps={FPS},setsar=1"
        run("ffmpeg", "-v", "error", "-y", "-ss", f"{s['a']:.3f}", "-i", SRC, "-frames:v", str(s["n"]), "-vf", vf,
            "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "17", "-pix_fmt", "yuv420p", f + ".tmp.mp4")
        os.rename(f + ".tmp.mp4", f)
        print(f"clip {s['i']+1}/{len(tl)}", flush=True)
    print("CLIPS_DONE", round(total, 2))

def audio():
    tl, total = timeline()
    parts, labels = [], []
    for s in tl:
        d = s["d"]
        parts.append(f"[0:a]atrim=start={s['a']:.4f}:duration={d:.4f},asetpts=PTS-STARTPTS,"
                     f"afade=t=in:d=0.015,afade=t=out:st={d-0.025:.4f}:d=0.025[a{s['i']}]")
        labels.append(f"[a{s['i']}]")
    parts.append(f"anullsrc=r=48000:cl=stereo,atrim=duration={END}[sil]"); labels.append("[sil]")
    chain = ("".join(labels) + f"concat=n={len(labels)}:v=0:a=1,pan=mono|c0=0.5*c0+0.5*c1,highpass=f=80,"
             f"arnndn=m={ASSETS}/sh.rnnn:mix=0.85,afftdn=nr=8:nf=-45,"
             "equalizer=f=200:t=q:w=1:g=-2,equalizer=f=3200:t=q:w=1.2:g=2.5,"
             "acompressor=threshold=-20dB:ratio=3:attack=8:release=120:makeup=3,"
             "loudnorm=I=-14:TP=-1.5:LRA=9,aresample=48000[out]")
    run("ffmpeg", "-v", "error", "-y", "-i", f"{WD}/audio_full.wav", "-filter_complex", ";".join(parts) + ";" + chain,
        "-map", "[out]", "-ac", "2", f"{WD}/audio_edit.wav")
    print("audio ok", round(total + END, 2))

def music():
    """Mix a licensed track the user sent (<work_dir>/music.* or cfg["music"]) under the voice.
    Constant bed ~music_db (default 25) dB below the voice, looped/trimmed, faded in/out.
    Idempotent: always starts from audio_voice.wav (voice-only copy made on first run)."""
    import shutil
    src = cfg.get("music") or next(iter(sorted(glob.glob(f"{WD}/music.*"))), None)
    if not src:
        print("MUSIC_SKIP: no track (put the user's file at <work_dir>/music.mp3)"); return
    voice, backup = f"{WD}/audio_edit.wav", f"{WD}/audio_voice.wav"
    if not os.path.exists(backup):
        shutil.copy(voice, backup)
    _, total = timeline(); dur = total + END
    lufs = -14 - cfg.get("music_db", 25)
    run("ffmpeg", "-v", "error", "-y", "-stream_loop", "-1", "-i", src, "-t", f"{dur:.3f}",
        "-af", f"aresample=48000,loudnorm=I={lufs}:TP=-9:LRA=7,afade=t=in:d=1.2,"
               f"afade=t=out:st={dur-2.5:.3f}:d=2.5", "-ac", "2", f"{WD}/music_bed.wav")
    run("ffmpeg", "-v", "error", "-y", "-i", backup, "-i", f"{WD}/music_bed.wav", "-filter_complex",
        "[0:a][1:a]amix=inputs=2:duration=first:normalize=0,alimiter=limit=0.84:level=0[out]",
        "-map", "[out]", "-ar", "48000", "-ac", "2", voice)
    print(f"music ok: bed at {lufs} LUFS (~{cfg.get('music_db', 25)} dB under the voice)")

RED, GREEN, WHITE, GRAY, DARK = "&H1306E3&", "&H99D334&", "&HFFFFFF&", "&H8A8A90&", "&H0D0B0B&"

def mk(s):  # [r]red[/r]  [g]green[/g]
    return (s.replace("[r]", f"{{\\1c{RED}}}").replace("[g]", f"{{\\1c{GREEN}}}")
             .replace("[/r]", f"{{\\1c{WHITE}}}").replace("[/g]", f"{{\\1c{WHITE}}}"))

def overlay():
    tl, total = timeline()
    Wd = json.load(open(f"{WD}/words.json"))
    fixes = {k.lower(): v for k, v in cfg.get("fixes", {}).items()}
    greens = [g.upper() for g in cfg.get("green_words", [])]
    ev = []
    def ts(t):
        t = max(0, t); return f"{int(t//3600)}:{int(t%3600//60):02d}:{t%60:05.2f}"
    def E(a, b, text, layer=1, style="Base"):
        ev.append(f"Dialogue: {layer},{ts(a)},{ts(b)},{style},,0,0,0,,{text}")
    def rect(x, y, w, h, color, alpha="&H00&", extra=""):
        return f"{{\\an7\\pos({x},{y})\\bord0\\shad0\\1c{color}\\1a{alpha}{extra}\\p1}}m 0 0 l {w} 0 {w} {h} 0 {h}{{\\p0}}"
    def txt(x, y, s, font, size, color=WHITE, an=7, extra=""):
        return f"{{\\an{an}\\pos({x},{y})\\fn{font}\\fs{size}\\1c{color}\\bord0\\shad0{extra}}}{mk(s)}"
    O = lambda t: src2out(tl, t)

    # subtitles: 3 words / 13 chars per block, active word red (or green for green_words), never overlapping
    hide = []
    cl = cfg.get("climax")
    if cl:
        cs = tl[cl["seg"]]; hide.append((cs["o"], O(cl["hide_subs_until"])))
    def clean(w):
        w = w.strip(); k = w.lower().strip(",.?!")
        if k in fixes:
            return fixes[k] + w[len(k):] if fixes[k] else ""
        return w
    for s in tl:
        ws = [dict(w=clean(x["w"]), s=x["s"], e=x["e"]) for x in Wd
              if x["s"] >= s["a"] - 0.05 and x["e"] <= s["b"] + 0.2 and x["s"] < s["b"] - 0.05]
        ws = [w for w in ws if w["w"]]
        chunks, cur = [], []
        for w in ws:
            cur.append(w)
            if len(cur) >= 3 or sum(len(c["w"]) for c in cur) > 13 or w["w"][-1] in ",.?!":
                chunks.append(cur); cur = []
        if cur: chunks.append(cur)
        for ci, ch in enumerate(chunks):
            for j, w in enumerate(ch):
                a = s["o"] + max(w["s"], s["a"]) - s["a"]
                nxt = ch[j + 1]["s"] if j + 1 < len(ch) else ch[-1]["e"] + 0.25
                if j + 1 == len(ch) and ci + 1 < len(chunks):
                    nxt = min(nxt, chunks[ci + 1][0]["s"])
                b = min(s["o"] + min(nxt, s["b"]) - s["a"], s["o"] + s["d"])
                if any(h0 <= a < h1 for h0, h1 in hide) or b <= a:
                    continue
                parts = []
                for k2, ww in enumerate(ch):
                    word = ww["w"].upper().rstrip(",.")
                    col = (GREEN if any(g in word for g in greens) else RED) if k2 == j else WHITE
                    parts.append(f"{{\\1c{col}}}{word}")
                pop = "\\fscx92\\fscy92\\t(0,90,\\fscx100\\fscy100)" if j == 0 else ""
                E(a, b, f"{{\\an2\\pos(540,1660){pop}}}" + "\\h\\h".join(parts), 5, "Sub")

    # act progress bar (top) + act opening card (chest zone, 2.4s)
    for ai, name in enumerate(cfg["acts"], 1):
        segs = [s for s in tl if s["act"] == ai]
        if not segs: continue
        a0, a1 = segs[0]["o"], segs[-1]["o"] + segs[-1]["d"]
        for k2 in range(len(cfg["acts"])):
            E(a0, a1, rect(70 + k2 * 104, 118, 90, 6, RED if k2 + 1 == ai else "&H403A3A&"), 2)
        E(a0, a1, txt(70, 136, f"ATO {ai} • {name}", "Barlow Condensed", 40, extra="\\b1\\fsp3"), 2)
        c1, fad = a0 + 2.4, "\\fad(220,260)"
        E(a0, c1, rect(0, 1040, 1080, 260, DARK, "&H40&", extra=fad), 3)
        E(a0, c1, rect(80, 1072, 10, 196, RED, extra=fad), 4)
        E(a0, c1, txt(118, 1066, f"ATO {ai}", "Barlow Condensed", 46, RED, extra=fad + "\\b1\\fsp6"), 4)
        first, *rest = name.split(" ", 1)
        E(a0, c1, txt(116, 1114, f"{first} [r]{rest[0] if rest else ''}[/r]", "Anton", 124, extra=fad), 4)

    for n in cfg.get("nametags", []):
        a = tl[n["seg"]]["o"] + 0.1; b = a + n.get("dur", 3.0); fad = "\\fad(200,250)"; role = n.get("role", "")
        E(a, b, rect(70, 1150, 9, 128 if role else 84, RED, extra=fad), 4)
        E(a, b, txt(98, 1136, n["name"], "Anton", 82, extra=fad + "\\bord3\\3c&H000000&"), 4)
        if role:
            E(a, b, txt(100, 1232, role, "Barlow Condensed", 46, extra=fad + "\\b1\\fsp2\\bord2\\3c&H000000&"), 4)

    fad = "\\fad(180,220)"
    for c in cfg.get("cards", []):
        a, b = O(c["from"]) + c.get("delay", 0), O(c["to"])
        if c["type"] == "stat":
            y, h = c.get("y", 1010), c.get("h", 300)
            E(a, b, rect(90, y, 900, h, DARK, "&H38&", extra=fad), 3)
            E(a, b, rect(90, y, 900, 5, RED, extra=fad), 4)
            E(a, b, txt(540, y + 24, c["big"], "Anton", c.get("size", 150), an=8, extra=fad), 4)
            E(a, b, txt(540, y + h - 32, c["label"], "Barlow Condensed", 42, RED, an=2, extra=fad + "\\b1\\fsp4"), 4)
        elif c["type"] == "list":   # big title + bullet items that appear when spoken
            n = len(c["items"]); h = 200 + n * 62 + 20
            E(a, b, rect(90, 960, 900, h, DARK, "&H38&", extra=fad), 3)
            E(a, b, rect(90, 960, 900, 5, RED, extra=fad), 4)
            E(a, b, txt(540, 980, c["title"], "Anton", 130, an=8, extra=fad), 4)
            for k2, (item, t) in enumerate(c["items"]):
                y = 1160 + k2 * 62
                E(O(t), b, rect(150, y + 14, 14, 14, RED, extra="\\fad(120,220)"), 4)
                E(O(t), b, txt(186, y, item, "Barlow Condensed", 50, extra="\\fad(120,220)\\b1\\fsp3"), 4)
        elif c["type"] == "grid":   # small header + up to 4 big words, 2 columns, appear when spoken
            E(a, b, rect(90, 990, 900, 380, DARK, "&H38&", extra=fad), 3)
            E(a, b, rect(90, 990, 900, 5, RED, extra=fad), 4)
            E(a, b, txt(140, 1012, c["header"], "Barlow Condensed", 42, RED, extra=fad + "\\b1\\fsp4"), 4)
            for k2, (word, t) in enumerate(c["items"]):
                E(O(t), b, txt(140 + (k2 % 2) * 420, 1080 + (k2 // 2) * 130, word, "Anton",
                               92 if len(word) <= 10 else 80, extra="\\fad(120,220)"), 4)

    if cl:
        cs = tl[cl["seg"]]; a, b = cs["o"], cs["o"] + cs["d"]; f0 = "\\fad(200,0)"
        L = cl["lines"]
        E(a, b, rect(0, 940, 1080, 560, DARK, "&H30&", extra=f0), 3)
        E(a, b, rect(80, 990, 10, 450, GREEN, extra=f0), 4)
        E(a, b, txt(120, 985, L[0], "Anton", 128, extra=f0), 4)
        E(a, b, txt(120, 1125, L[1], "Anton", 96, extra=f0), 4)
        E(O(cl["line3_at"]), b, txt(120, 1240, L[2], "Anton", 100,
          extra="\\fad(160,0)\\fscx90\\fscy90\\t(0,140,\\fscx100\\fscy100)"), 4)

    e = cfg["end"]; t0, t1 = total, total + END; f0, f1 = "\\fad(350,0)", "\\fad(300,0)"
    E(t0, t1, rect(100, 700, 10, 330, RED, extra=f0), 4)
    E(t0, t1, txt(140, 690, e["eyebrow"], "Barlow Condensed", 44, RED, extra=f0 + "\\b1\\fsp5"), 4)
    E(t0, t1, txt(136, 745, e["title"], "Anton", 140, extra=f0), 4)
    E(t0, t1, txt(140, 935, e.get("subtitle", ""), "Barlow Condensed", 62, extra=f0 + "\\b1\\fsp8"), 4)
    if e.get("url"):
        E(t0 + 0.4, t1, rect(140, 1130, 800, 120, "&H2B2626&", extra=f1), 3)
        E(t0 + 0.4, t1, rect(142, 1132, 796, 116, "&H161313&", extra=f1), 3)
        E(t0 + 0.4, t1, txt(540, 1190, e["url"], "Montserrat", 58, an=5, extra=f1 + "\\b1"), 4)
    if e.get("footer"):
        E(t0 + 0.7, t1, txt(540, 1290, e["footer"], "Barlow Condensed", 38, GRAY, an=8, extra=f1 + "\\b1\\fsp3"), 4)

    head = ("[Script Info]\nScriptType: v4.00+\nPlayResX: 1080\nPlayResY: 1920\nWrapStyle: 2\nScaledBorderAndShadow: yes\n\n"
            "[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, "
            "Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
            "Style: Base,Montserrat,48,&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,0,0,7,0,0,0,1\n"
            "Style: Sub,Montserrat Black,82,&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,6,0,2,60,60,0,1\n\n"
            "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")
    open(f"{WD}/overlay.ass", "w").write(head + "\n".join(ev) + "\n")
    # sanity: subtitle overlaps + chest-zone (layer 3 boxes) collisions
    def tsec(x): return sum(float(p) * m for p, m in zip(x.split(":"), [3600, 60, 1]))
    subs = sorted((tsec(l.split(",")[1]), tsec(l.split(",")[2])) for l in ev if l.startswith("Dialogue: 5"))
    boxes = sorted((tsec(l.split(",")[1]), tsec(l.split(",")[2])) for l in ev if l.startswith("Dialogue: 3") and "\\1a" in l and "pos(0,9" not in l and "pos(14" not in l)
    ov = sum(1 for x, y in zip(subs, subs[1:]) if y[0] < x[1] - 0.001)
    col = [(round(x[0], 1), round(y[0], 1)) for x, y in zip(boxes, boxes[1:]) if y[0] < x[1] - 0.05]
    print(f"overlay ok: {len(ev)} events, video {total + END:.1f}s, subtitle overlaps={ov}, card collisions={col}")

def compose():
    tl, total = timeline()
    out = f"{WD}/{cfg.get('out_name', 'video_9x16')}.mp4"
    import numpy as np
    from PIL import Image
    Wb, Hb = 1080, 1920
    y, x = np.mgrid[0:Hb, 0:Wb].astype(np.float32)
    d = np.sqrt(((x - 0.75 * Wb) / Wb) ** 2 + ((y - 0.45 * Hb) / Hb) ** 2) / 0.9
    stops = [(0, (0x3A, 0x0A, 0x0E)), (0.40, (0x14, 0x14, 0x17)), (1.0, (0x0B, 0x0B, 0x0D))]
    img = np.stack([np.interp(np.clip(d, 0, 1), [s[0] for s in stops], [s[1][c] for s in stops]) for c in range(3)], -1)
    img += np.random.uniform(-1.2, 1.2, img.shape)
    Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(f"{WD}/endbg.png")
    run("ffmpeg", "-v", "error", "-y", "-loop", "1", "-framerate", str(FPS), "-i", f"{WD}/endbg.png",
        "-frames:v", str(round(END * FPS)), "-vf", "format=yuv420p,setsar=1", "-c:v", "libx264", "-preset", "veryfast",
        "-crf", "17", f"{WD}/clips/end.mp4")
    with open(f"{WD}/clips/list.txt", "w") as f:
        for s in tl:
            f.write(f"file '{WD}/clips/c{s['i']:02d}.mp4'\n")
        f.write(f"file '{WD}/clips/end.mp4'\n")
    run("ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", f"{WD}/clips/list.txt", "-c", "copy", f"{WD}/base.mp4")
    vf = (f"eq=contrast=1.05:saturation=1.06,subtitles={WD}/overlay.ass:fontsdir={FONTS},"
          f"fade=t=out:st={total + END - 0.4:.3f}:d=0.4")
    run("ffmpeg", "-v", "error", "-y", "-i", f"{WD}/base.mp4", "-i", f"{WD}/audio_edit.wav", "-map", "0:v", "-map", "1:a",
        "-vf", vf, "-c:v", "libx264", "-preset", "veryfast", "-crf", "19", "-pix_fmt", "yuv420p", "-profile:v", "high",
        "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", out)
    print("FINAL_OK", out)

def qa():
    from PIL import Image
    tl, total = timeline()
    out = f"{WD}/{cfg.get('out_name', 'video_9x16')}.mp4"
    marks = [1.5] + [next(s["o"] for s in tl if s["act"] == a) + 1.2 for a in range(1, len(cfg["acts"]) + 1)]
    marks += [src2out(tl, (c["from"] + c["to"]) / 2) for c in cfg.get("cards", [])]
    if cfg.get("climax"):
        marks.append(tl[cfg["climax"]["seg"]]["o"] + tl[cfg["climax"]["seg"]]["d"] - 0.6)
    marks.append(total + END - 0.8)
    marks = sorted(marks)[:16]
    ims = []
    for i, t in enumerate(marks):
        p = f"{WD}/qa_{i:02d}.png"
        run("ffmpeg", "-v", "error", "-y", "-ss", f"{t:.2f}", "-i", out, "-frames:v", "1", "-vf", "scale=360:640", p)
        ims.append(Image.open(p))
    for r in range(0, len(ims), 8):
        row = ims[r:r + 8]; g = Image.new("RGB", (360 * len(row), 640))
        for i, im in enumerate(row): g.paste(im, (i * 360, 0))
        g.save(f"{WD}/qa_sheet_{r // 8}.jpg", quality=85); print(f"{WD}/qa_sheet_{r // 8}.jpg")
    r = subprocess.run(["ffmpeg", "-hide_banner", "-i", out, "-af", "ebur128=peak=true", "-f", "null", "-"],
                       capture_output=True, text=True).stderr
    print([l.strip() for l in r.splitlines() if l.strip().startswith(("I:", "Peak:"))][-2:])

def light():
    out = f"{WD}/{cfg.get('out_name', 'video_9x16')}.mp4"
    run("ffmpeg", "-v", "error", "-y", "-i", out, "-c:v", "libx264", "-preset", "veryfast", "-crf", "25",
        "-maxrate", "3M", "-bufsize", "6M", "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart",
        out.replace(".mp4", "_leve.mp4"))
    print("LIGHT_OK", out.replace(".mp4", "_leve.mp4"))

{"setup": setup, "transcribe": transcribe, "faces": faces, "cropcheck": cropcheck, "clips": clips, "audio": audio, "music": music,
 "overlay": overlay, "compose": compose, "qa": qa, "light": light}[STEP]()
