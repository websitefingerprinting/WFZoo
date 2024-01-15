# WFZoo： A Toolbox for SOTA Website Fingerprinting Attacks and Defenses
Since my previous toolbox is not well organized and out-of-date (written in keras and py37), I decide to rewrite it. This toolbox is still under development. I will add more attacks and defenses in the future.
I tested the code under both Python 3.8 and Python 3.11. The conda environment I provided is based on Python 3.8 with pytorch 2.0 and cuda 11.

## TODO List
- About attacks
    - [x] DF attack
    - [x] Tik-Tok attack
    - [ ] Add more attacks
    - [x] Implement Pytorch ``amp`` mode
- About defenses
    - [x] FRONT
    - [x] Tamaraw

## How to use
### 1. Install dependencies

```bash
conda env create -f py38.yaml
conda activate py38
```

### 2. Dataset Format
It has been a convention to name a trace as "A-B.cell" or "C.cell". 
Here, A is the class number of the trace, B is the instance number of the trace. 
"C.cell" is the C-th non-monitored trace in the dataset.

### 3. Train and Test an Attack
```bash
python run_attack.py  --attack df --data-path ~/somedataset/ \
--mon-classes 100 --mon-inst 100 --unmon-inst 10000 \
--open-world
```
This command will perform a 10-cross-validation attack on the given dataset. 

``--attack`` specifies the attack to evaluate (currently support DF and Tik-Tok).

``--data-path`` specifies the path of the dataset.

``--mon-classes`` specifies the number of monitored classes.

``--mon-inst`` specifies the number of monitored instances per class.

``--unmon-inst`` specifies the number of unmonitored instances.

``--open-world`` makes an open-world evaluation (by default: closed-world evaluation)

``--one-fold`` only run one fold instead of 10 cross validation.

``--suffix`` specifies the suffix of each file in the dataset (By default: `.cell`). 
Change it if your dataset is not end with `.cell`.

### 4. Simulate a Defense
```bash
python run_defense.py --defense front --data-path ~/somedataset/ \
--config-path ./defenses/config/front.ini \
--mon-classes 100 --mon-inst 100 --unmon-inst 10000 --open-world
```