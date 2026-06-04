# Stage_Python_robot

Projet de stage ESIGELEC visant à développer une plateforme d’apprentissage et d’expérimentation en intelligence artificielle, Python et robotique autonome.

## document lien https://www.stereolabs.com/docs/embedded/zed-box
## demo lien https://github.com/stereolabs/zed-sdk/tree/master
## SDK download: https://www.stereolabs.com/en-fr/developers/release
## YOLO document:https://docs.ultralytics.com/#where-to-start
## Membres

- Ran Pan – Détection des situations d'urgence
- Zhen Yang – Navigation autonome entre A et B
- Liu Dongtao – Détection d'obstacles en temps réel

## Objectifs

1. Développer les compétences en IA et Python.
2. Construire une plateforme pédagogique pour les TP de l'ESIGELEC.
3. Optimiser les modules et favoriser leur interopérabilité.




-----------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------
## Configuration du matériel : ZED Box
Démarrage du ZED Box

1. Alimentation
Connecter l’alimentation du ZED Box.
Après la mise sous tension, le voyant vert (Power Status) doit s’allumer, indiquant que le système est alimenté correctement.

2. Connexion de l’écran
Connecter un écran au port HDMI du ZED Box afin d’accéder à l’interface Ubuntu.

3. Connexion réseau
Connecter un câble Ethernet entre le ZED Box et l’ordinateur (ou le réseau local) pour permettre la communication et l’accès à distance via SSH.

4. Clavier et souris
Brancher un clavier et une souris sur les ports USB du ZED Box.

5. Démarrage
Une fois toutes les connexions effectuées, démarrer le système et se connecter à Ubuntu.


## Configuration matérielle

### Plateforme embarquée

Le projet est développé sur une plateforme embarquée **Stereolabs ZED Box** intégrant :

- NVIDIA Jetson Orin NX
- Architecture ARM64 (aarch64)
- NVIDIA JetPack 6.0
- Ubuntu 22.04.4 LTS
- CUDA 12.2
- ZED SDK préinstallé

### Caméra utilisée

- Stereolabs ZED 2i
- Caméra stéréoscopique (vision binoculaire)
- Acquisition RGB et profondeur (Depth)
- Compatible avec le SDK ZED pour la détection d'obstacles en temps réel

### Logiciels installés

Le système dispose des outils suivants :

- ZED Explorer
- ZED Depth Viewer
- ZED Calibration
- ZED Diagnostic
- ZED Sensor Viewer
- ZEDfu
- ZED Media Server

### Informations système

| Élément | Valeur |
|----------|----------|
| Plateforme | Stereolabs ZED Box |
| Module GPU | NVIDIA Jetson Orin NX |
| Architecture | ARM64 (aarch64) |
| Système | Ubuntu 22.04.4 LTS |
| JetPack | 6.0 |
| L4T | R36.3 |
| Python | 3.10.12 |
| CUDA | 12.2 |
| SDK Vision | ZED SDK |

### Informations de connexion par défaut

Pour la configuration initiale du ZED Box :

Nom d'utilisateur : user
Mot de passe : admin

## IDE Configuration (VS Code)

### Why VS Code

The project is mainly developed in Python and will later integrate:

* ZED SDK
* OpenCV
* YOLO
* ROS2

Visual Studio Code is recommended because it provides:

* Python development support
* Integrated terminal
* Git integration
* Remote development capabilities
* ROS2 extensions
* YOLO/OpenCV development support

---

### Install VS Code on Jetson (ARM64)

#### 1. Install required tools

```bash
sudo apt install wget gpg
```

#### 2. Import Microsoft's signing key

```bash
wget -qO- https://packages.microsoft.com/keys/microsoft.asc | sudo gpg --dearmor -o /usr/share/keyrings/microsoft.gpg
```

#### 3. Add the Microsoft repository

```bash
echo "Types: deb
URIs: https://packages.microsoft.com/repos/code
Suites: stable
Components: main
Architectures: amd64,arm64,armhf
Signed-By: /usr/share/keyrings/microsoft.gpg" | sudo tee /etc/apt/sources.list.d/vscode.sources
```

#### 4. Update package list

```bash
sudo apt update
```

#### 5. Install VS Code

```bash
sudo apt install code
```

#### 6. Launch VS Code

```bash
code
```

or from the Ubuntu application menu.

---

### Notes

The Jetson platform uses the ARM64 architecture (`aarch64`).

Verify the architecture using:
## VS Code – Création du trousseau de clés (Keyring)

Lors de la première connexion à GitHub ou GitHub Copilot dans VS Code sous Linux, une fenêtre peut apparaître pour créer un **Keyring** (« Default Keyring »).

### Solution

Saisissez votre **mot de passe de session Linux**, puis cliquez sur **Continue**.

### Pourquoi ?

Le Keyring permet de stocker de manière sécurisée les identifiants et jetons d’authentification utilisés par VS Code, GitHub et GitHub Copilot.

### Remarque

Si vous cliquez sur **Cancel**, VS Code risque de ne pas enregistrer vos informations de connexion.




-----------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------


