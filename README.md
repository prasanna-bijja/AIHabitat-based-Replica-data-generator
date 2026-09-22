# AIHabitat-based-Replica-data-generator



Clone Habitat-Sim:

```bash
git clone --recursive https://github.com/facebookresearch/habitat-sim.git
cd habitat-sim
```

- If the repository was cloned without submodules:

- git submodule update --init --recursive

- Install the required Python packages:
```bash
pip install numpy
pip install scikit-build-core
```

Build Habitat-Sim:
```bash
pip install . --no-build-isolation
Download Replica Dataset
```
Clone the official Replica repository:

```bash
git clone https://github.com/facebookresearch/Replica-Dataset.git
cd Replica-Dataset
```

Install the download dependencies:
```bash
sudo apt-get install wget pigz unzip
```
Download the dataset:
```bash
./download.sh /path/to/replica
```
For example:
```bash
./download.sh ~/habitat-sim/data/scene_datasets/replica
```
Data Recording

The script allows interactive navigation inside a Replica scene while
recording synchronized sensor observations.

```bash
python record_navigation.py
```

* Navigation:

W/A/S/D – move camera
I/J/K/L - turn camera
Z/X – move up/down
Arrow keys – rotate camera
Space key – save synchronized frame
Output

For each recorded frame, the following information is stored:

```bash output/
├── rgb/
│   └── 000000.png
├── depth/
│   └── 000000.npy
├── semantic/
│   └── 000000.npy
├── instance/
│   └── 000000.npy
├── poses/
│   └── 000000.npy
├── intrinsics.txt
├── trajectory.txt
└── frames.csv



trajectory.txt follows:

timestamp tx ty tz qx qy qz qw
