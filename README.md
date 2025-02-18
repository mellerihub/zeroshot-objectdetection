
## :hammer_and_wrench:  Requirements and Install 

Basic Dependencies:

Python == 3.10
Pytorch >= 2.1.0
CUDA Version >= 11.7

** Installation:**

<br>
1.Clone the ALO repository from GitHub.

```bash
git clone https://github.com/mellerikat/alo.git zeroshot_od
```

<br>
2. Clone the zeroshot-objectdetection solution repository from GitHub.

```bash
git clone https://github.com/mellerihub/zeroshot-objectdetection.git
```

<br>
3. Download pre=trained tokenizer adn model weights.

```bash
cd assets
cd inference
git clone git@hf.co:google-bert/bert-base-uncased

cd groundingdino
cd checkpoints
wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
```

<br>
4. Install the required dependencies in the solution directory.

```bash
pip install -r requirements.txt
cd assets
cd inference
pip install -e .
```

<br>
5. Local Demo start

```bash
python main.py
```
