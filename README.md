# Radio Silence

An AI rebuilds brain MRI scans from only part of the data, which would make scans faster.
This project shows **where the AI is guessing**, so a doctor knows which parts of the picture
to double-check.

## What you need

- A computer with Python 3.10 or newer
- An NVIDIA graphics card (recommended; without one it still works, just much slower)
- The fastMRI brain `.h5` scan files (ask the team; they can't be shared publicly)

## How to run it

**1. Download the code**

```bash
git clone https://github.com/AbdelrahmanYousssef/RadioSIlence.git
cd RadioSIlence
```

**2. Download the two helper repos it uses**

```bash
git clone https://github.com/facebookresearch/fastMRI third_party/fastMRI
git clone https://github.com/microsoft/fastmri-plus third_party/fastmri-plus
```

**3. Install the Python packages**

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

**4. Add the scans**

Make a folder called `DATA` inside `RadioSIlence` and put the `.h5` files in it.

**5. Start the app**

```bash
python app.py
```

Wait until it says `Open http://127.0.0.1:5000`, then open that link in your browser.
The first start takes a few minutes, because it downloads the AI model (120 MB).

To stop it, press `Ctrl + C` in the terminal.

## Using the app

You'll see three pictures:

1. **Real scan**: the normal, slow scan
2. **AI's version**: the AI's picture, made from only about a quarter of the data
3. **Where the AI is guessing**: bright yellow means the AI is unsure

Blue squares are lesions marked by a radiologist.

- **Patient 1 / 2 / 3** switches scans
- **‹ ›** (or the arrow keys) goes to the next picture
- **4× / 8× faster scan** throws away more data, and the guessing grows

Each new picture takes about 10 seconds.

## Something went wrong?

- **"no .h5 files"**: the scans aren't in the `DATA` folder.
- **Very slow**: no NVIDIA graphics card was found, so it's running on the processor.
- **`No module named ...`**: run step 3 again.

## For the team

- The **Full research view** button (top right of the app) has the detailed version.
  Its "Study results" tab needs you to run `python trustmap.py --accel 4 8` once first.
- [DETAILS.md](DETAILS.md) has the method, what was checked, and what was assumed.
- Never commit the scans, or images made from them, to GitHub (fastMRI data agreement).
  The `.gitignore` already blocks them.
