# Stage_Python_robot

Projet de stage ESIGELEC visant à développer une plateforme d’apprentissage et d’expérimentation en intelligence artificielle, Python et robotique autonome.

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


-----------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------


