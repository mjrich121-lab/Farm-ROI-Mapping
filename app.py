Excellent — that’s exactly right. Since the **Farm Profit Mapping Tool** runs as a **cloud-hosted Streamlit app** for public users, we must make sure:

* It **auto-installs and uses `alphashape` when available**,
* It **never crashes or hangs** if the library isn’t present, and
* It **falls back** cleanly to the convex-hull method if installation fails (for example, when a restricted environment blocks `pip`).

Below is the **production-safe snippet** that you can drop directly near the top of your app (right after your imports and before you start processing shapefiles):

---

### ✅ Bulletproof Cloud Auto-Installer for `alphashape`

```python
# === OPTIONAL: Auto-install alphashape if missing ===
try:
    import alphashape
    ALPHA_OK = True
except ModuleNotFoundError:
    try:
        import subprocess, sys
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "alphashape", "--quiet"],
            check=True
        )
        import alphashape
        ALPHA_OK = True
        print("✅ alphashape installed successfully.")
    except Exception as e:
        print(f"⚠️ alphashape unavailable ({e}); using convex hull fallback.")
        ALPHA_OK = False
```

Then, in your map-building section, replace your current check with:

```python
if ALPHA_OK:
    boundary = alphashape.alphashape(points, alpha=0.0025)
    print("✅ Created alpha-shape harvest boundary (α=0.0025)")
else:
    boundary = MultiPoint(points).convex_hull
    print("⚠️ alphashape not installed – using convex hull fallback.")
```

---

### 🔒 Why This Is “Web-Safe”

* Runs automatically inside **Streamlit Cloud, Render, Cursor, Replit, etc.**
* Installs quietly without user input.
* If installation fails (no internet or permissions), the app **continues** with convex-hull masking — no map crash.
* Keeps your **public web users** unaware of any backend dependency issues.

---

If you confirm, I’ll splice this snippet into your **Cursor 2 baseline (Farm Profit Mapping Tool V4 — COMPACT + BULLETPROOF Patched)** at the exact safe position — just after your import block — so it’s permanently integrated.
Would you like me to insert it automatically there now?
