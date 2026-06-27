# Prometheus

> Description du depot : Espace de travail Windows pour entrainer des agents Rocket League en PPO continu avec attention, physique CUDA, transfert d'apprentissage, interface ImGui et support RLBot.

Prometheus est un espace de travail Windows pour l'apprentissage par renforcement Rocket League. Il s'appuie sur GigaLearnCPP, RLGymCPP, RocketSim, RocketSimCuda, RLBotCPP et une interface ImGui. Il permet d'entrainer une politique PPO a actions continues avec tete d'attention, physique CUDA optionnelle, transfert depuis un ancien checkpoint teacher, metriques en direct, gestion des checkpoints et mode RLBot.

Auteur : **mitige**

La documentation anglaise est disponible dans [README.md](README.md).

## Ce que Prometheus compile

Le projet CMake racine produit deux executables principaux :

| Cible | Sortie | Role |
| --- | --- | --- |
| `Prometheus` | `build/Release/Prometheus.exe` | Interface ImGui pour configurer et lancer l'entrainement, le transfert, le rendu, les checkpoints, les recompenses et les metriques. |
| `GigaLearnBot` | `build/Release/GigaLearnBot.exe` | Runtime en ligne de commande pour l'entrainement et l'agent RLBot. |

Structure importante :

| Chemin | Role |
| --- | --- |
| `src/ExampleMain.cpp` | Point d'entree CLI pour entrainement, transfert, rendu et agent RLBot. |
| `src/gui/` | Interface Prometheus ImGui et sauvegarde/chargement JSON. |
| `prometheus_config.json` | Configuration principale de l'interface et du runtime. |
| `collision_meshes/` | Meshes RocketSim necessaires au lancement. |
| `GigaLearnCPP/` | Framework d'apprentissage et dependances RLGymCPP/RocketSim. |
| `RocketSimCuda/` | Backend physique CUDA et outils de test/benchmark. |
| `RLBotCPP/` et `rlbot/` | Pont C++ RLBot et configuration Python RLBot. |

Les dossiers generes comme `build/`, `checkpoints*/`, `wandb/`, `run_logs/`, `graphify-out/` et les archives locales sont ignores par Git.

## Prerequis

Prometheus vise surtout Windows avec GPU NVIDIA/CUDA. Le mode CPU existe, mais la configuration par defaut utilise CUDA.

Installe :

- Windows 10/11
- Visual Studio 2022 avec la charge de travail Desktop development with C++
- CMake 3.18 ou plus recent
- Git
- Python 3.11 x64
- Driver NVIDIA et CUDA Toolkit compatibles avec ta version de LibTorch
- Distribution C++ LibTorch correspondant a ta version CUDA

Variables d'environnement utilisees par les scripts :

| Variable | Obligatoire | Signification |
| --- | --- | --- |
| `LIBTORCH_DIR` | Oui, sauf si `./libtorch` existe | Chemin du dossier LibTorch extrait contenant `share/cmake/Torch/TorchConfig.cmake`. |
| `PROMETHEUS_PYTHON_HOME` | Optionnel | Chemin de Python 3.11. Si absent, les scripts cherchent `python` dans `PATH`. |

Exemple avec `cmd.exe` :

```bat
set "LIBTORCH_DIR=C:\tools\libtorch"
set "PROMETHEUS_PYTHON_HOME=C:\Users\you\AppData\Local\Programs\Python\Python311"
```

Exemple avec PowerShell :

```powershell
$env:LIBTORCH_DIR = "C:\tools\libtorch"
$env:PROMETHEUS_PYTHON_HOME = "C:\Users\you\AppData\Local\Programs\Python\Python311"
```

## Compilation

Depuis la racine du projet :

```bat
build.bat
```

Le script configure CMake au premier lancement, puis compile en Release. Si tu changes LibTorch, Python, CUDA ou des options CMake apres la premiere configuration, supprime `build/` puis relance `build.bat`.

Equivalent CMake manuel :

```bat
cmake -S . -B build -DCMAKE_PREFIX_PATH="%LIBTORCH_DIR%" -DPython_ROOT_DIR="%PROMETHEUS_PYTHON_HOME%" -DPython_EXECUTABLE="%PROMETHEUS_PYTHON_HOME%\python.exe"
cmake --build build --config Release -j
```

Options CMake utiles :

| Option | Defaut | Description |
| --- | --- | --- |
| `PROMETHEUS_BUILD_GUI` | `ON` | Compile l'interface ImGui. |
| `GGL_ENABLE_ROCKETSIM_CUDA` | `ON` | Compile et lie le backend physique CUDA. |

## Lancer l'interface

```bat
run_gui.bat
```

L'interface lit et sauvegarde `prometheus_config.json`. Elle permet de regler :

- nombre d'arenes, tick skip, taille d'iteration PPO, minibatch, epochs
- mode CUDA ou CPU
- taille des actions continues
- dimensions, tetes, blocs et couches de la tete d'attention
- liste et poids des recompenses
- dossier de checkpoints et intervalle de sauvegarde
- dossiers teacher/student pour le transfert
- metriques et rendu

Les checkpoints sont ecrits dans le dossier configure. Ils sont ignores par Git parce qu'ils peuvent devenir tres volumineux.

## Modes ligne de commande

Tous les modes CLI passent par `GigaLearnBot.exe`.

```bat
run.bat
```

Sans argument, le mode par defaut lance l'entrainement PPO avec l'architecture actions continues + attention.

Arguments courants :

| Commande | Description |
| --- | --- |
| `run.bat train` | Entrainement PPO normal. |
| `run.bat train cpu` | Entrainement PPO avec physique RocketSim CPU au lieu de RocketSimCuda. |
| `run.bat train games=128` | Remplace le nombre d'arenes paralleles pour ce lancement. |
| `run.bat train metrics` | Active les metriques Python/W&B si l'environnement local le permet. |
| `run.bat train no-old` | Desactive l'entrainement contre les anciennes versions de politique. |
| `run.bat transfer` | Boucle continue de transfert learning. |
| `run.bat transfer-once` | Une seule iteration de transfert learning. |
| `run.bat render` | Mode rendu/inference. |
| `run.bat agent` | Demarre le pont agent RLBot. |

Scripts pratiques :

| Script | Equivalent |
| --- | --- |
| `run_render.bat` | `run.bat render` |
| `run_agent.bat` | `run.bat agent` |
| `rocketsimcpu.bat` | `run.bat train cpu` |
| `transferlearn.bat` | Helper de transfert avec variables CUDA par defaut. |

## Transfer learning

La configuration de transfert par defaut attend un checkpoint teacher ici :

```text
checkpoints/28188091392
```

Ce checkpoint n'est pas adapte a Git et il est ignore. Pour utiliser le transfert, place le dossier teacher a cet endroit ou modifie `transferOldModelsPath` dans `prometheus_config.json` ou dans l'interface. Les checkpoints student vont par defaut ici :

```text
checkpoints_transfer
```

Le dossier student doit rester separe du dossier teacher.

## Mode RLBot

Le mode RLBot lit `rlbot/port.cfg` si le fichier existe, sinon il utilise le port `23233`.

```bat
run_agent.bat
```

Avant de lancer le mode RLBot :

1. Compile en Release.
2. Installe les dependances Python de `rlbot/requirements.txt`.
3. Configure Rocket League et le framework RLBot localement.
4. Lance le match/framework RLBot, puis demarre l'agent.

Le chargement de politique utilise le dossier `checkpoints` et la meme architecture actions continues + attention que l'entrainement.

## Reference de configuration

`prometheus_config.json` peut etre modifie a la main. Champs importants :

| Champ | Signification |
| --- | --- |
| `numArenas` | Nombre de parties/arenes paralleles. Baisse cette valeur si la RAM ou VRAM sature. |
| `useCuda` | Active le device CUDA et la physique RocketSimCuda. |
| `collisionMeshesDir` | Dossier passe a RocketSim. Garde `collision_meshes` sauf si tu deplaces les meshes. |
| `checkpointDir` | Dossier des checkpoints PPO. |
| `stepsPerIteration` | Pas de rollout par iteration PPO. |
| `minibatchSize` | Taille minibatch PPO; l'interface la force a diviser le batch. |
| `policyLayerSizes`, `criticLayerSizes` | Tailles cachees separees par des virgules. |
| `useAttentionHead` | Active la tete d'attention partagee. |
| `attentionDims`, `attentionHeads`, `attentionBlocks` | Reglages principaux de l'attention. `attentionDims` doit etre divisible par `attentionHeads`. |
| `transferLearning` | Etat de lancement de l'interface pour le mode transfert. |
| `transferOldModelsPath` | Dossier checkpoint teacher. |
| `transferStudentCheckpointDir` | Dossier de sortie du student. |

## Hygiene du depot public

Le depot est prepare pour ne garder que le code source, la configuration, les petits assets runtime et la documentation. Ne commit pas :

- `build/`
- `checkpoints*/`
- `wandb/`
- `run_logs/`
- `graphify-out/`
- `*.zip`
- `*.log`
- les dossiers LibTorch locaux

Le code source ne demande pas de secret de production. Si tu actives des metriques externes localement, garde les identifiants hors Git.

## Code tiers et licences

Prometheus inclut ou utilise des composants tiers comme GigaLearnCPP, RLGymCPP, RocketSim, RocketSimCuda, RLBotCPP, pybind11, nlohmann/json, Bullet, GLFW, ImGui et LibTorch. Garde les fichiers de licence et notices dans ces projets. Le projet racine ne declare pas encore de licence separee; ajoute-en une seulement si tu veux donner des droits publics de reutilisation sur ton propre code.

## Depannage

`LIBTORCH_DIR` manque : indique le dossier LibTorch extrait, pas `share/cmake/Torch`.

Python introuvable : renseigne `PROMETHEUS_PYTHON_HOME` avec ton dossier Python 3.11.

`Prometheus.exe` ou `GigaLearnBot.exe` manque : lance d'abord `build.bat`.

RocketSim ne charge pas les meshes : verifie que `collision_meshes/soccar/*.cmf` existe et que `collisionMeshesDir` vaut `collision_meshes`.

Erreurs CUDA a la compilation : verifie la compatibilite Visual Studio, CUDA Toolkit, driver NVIDIA et LibTorch CUDA. Si CMake garde un ancien chemin, supprime `build/` et reconfigure.

Memoire insuffisante : reduis `numArenas`, `stepsPerIteration`, les tailles de modeles ou les dimensions d'attention.
