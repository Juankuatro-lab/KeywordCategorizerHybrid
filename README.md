# 🎯 Script de catégorisation de mots-clés

Un outil intelligent de catégorisation automatique de mots-clés utilisant l'Intelligence Artificielle, développé avec Streamlit.

## 📋 Description

Cet outil utilise l'Intelligence Artificielle pour catégoriser automatiquement des mots-clés en préservant exactement la structure de votre fichier d'entrée. Il combine plusieurs techniques NLP pour une catégorisation précise et contextuelle avec des options de performance adaptées à vos besoins.

## ✨ Fonctionnalités principales

- **🔄 Préservation de structure** : Votre fichier garde exactement sa mise en forme originale
- **🤖 IA avancée** : Utilise des modèles de langue de pointe pour comprendre le contexte
- **⚡ Performance optimisée** : Trois modes de traitement selon vos besoins
- **🌍 Multilingue** : Fonctionne en français, anglais et autres langues européennes
- **📊 Attribution forcée** : Chaque mot-clé est obligatoirement attribué à une catégorie
- **📈 Statistiques détaillées** : Scores de confiance et analyses complètes

## 🛠 Technologies utilisées

- **Embeddings sémantiques** : Modèle Sentence-BERT multilingue
- **Similarité cosinus** : Mesure de proximité sémantique dans l'espace vectoriel
- **TF-IDF** : Analyse de fréquence des termes
- **Algorithme hybride** : Combinaison de 3 approches pour maximiser la précision
- **Traitement par batch** : Optimisation pour les gros volumes

## 🚀 Modes de traitement

| Mode | Vitesse | Précision | Recommandé pour |
|------|---------|-----------|-----------------|
| 🚀 **Rapide** | ~1000x plus rapide | Bonne | >1000 lignes, tests rapides |
| 🎯 **Précis** | Plus lent | Excellente | <1000 lignes, qualité maximale |
| ⚡ **Auto** | Adaptatif | Optimale | Tous usages |

## 📊 Performance estimée

| Nombre de lignes | Mode rapide | Mode IA | Mode batch |
|------------------|-------------|---------|------------|
| 100 lignes       | <1 sec      | ~5 sec  | ~2 sec     |
| 1,000 lignes     | ~1 sec      | ~50 sec | ~10 sec    |
| 10,000 lignes    | ~10 sec     | ~8 min  | ~2 min     |

## 📦 Installation

### Prérequis
- Python 3.8+
- pip ou conda

### Installation des dépendances

#### Version minimale (mode rapide uniquement)
```bash
pip install streamlit pandas openpyxl chardet numpy
```

#### Version complète (avec IA avancée)
```bash
pip install -r requirements.txt
```

### Contenu du fichier requirements.txt
```
streamlit>=1.28.0
pandas>=1.5.0
openpyxl>=3.0.0
chardet>=5.0.0
numpy>=1.21.0
sentence-transformers>=2.2.0
scikit-learn>=1.3.0
transformers>=4.21.0
torch>=1.13.0
```

## 🏃‍♂️ Utilisation

### Lancement local
```bash
streamlit run keyword_categorizer.py
```

### Déploiement sur Streamlit Cloud
1. Forkez ce repository
2. Connectez votre compte GitHub à Streamlit Cloud
3. Déployez l'application en sélectionnant `keyword_categorizer.py`

## 📝 Guide d'utilisation

### Étape 1 : Configuration des catégories
1. Créez vos catégories (ex: "Produits", "Services", "Marques")
2. Ajoutez les termes de référence pour chaque catégorie
3. Gérez vos catégories (ajout/suppression)

### Étape 2 : Import du fichier
1. Préparez votre fichier Excel ou CSV
2. Assurez-vous qu'il contient :
   - Une colonne avec vos mots-clés
   - Des colonnes pour les résultats de chaque catégorie
   - Toute autre donnée à conserver

### Étape 3 : Configuration des colonnes
1. Sélectionnez la colonne des mots-clés
2. Assignez une colonne de sortie pour chaque catégorie
3. Vérifiez la configuration

### Étape 4 : Choix du mode de traitement
1. **Mode Rapide** : Pour les gros fichiers ou tests rapides
2. **Mode Précis** : Pour la meilleure qualité de catégorisation
3. **Mode Auto** : Choix intelligent selon la taille du fichier
4. Activez le traitement par batch si vous avez beaucoup de doublons

### Étape 5 : Traitement et export
1. Lancez la catégorisation
2. Consultez les statistiques
3. Exportez vos résultats (Excel/CSV)

## 📈 Scores de confiance

- **0.8-1.0** : Correspondance excellente (IA très confiante)
- **0.6-0.8** : Correspondance bonne (contexte sémantique fort)
- **0.4-0.6** : Correspondance acceptable (similarité détectée)
- **0.0-0.4** : Correspondance faible (attribution par défaut)

## 💡 Recommandations d'utilisation

- **Fichiers <1000 lignes** : Utilisez le mode Auto ou Précis
- **Fichiers >1000 lignes** : Commencez par le mode Rapide pour te
