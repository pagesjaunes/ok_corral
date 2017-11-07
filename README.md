### API Décisions: Ok Corral

Api pour la prise de décision en ligne à information partielle (pour faire du bandit quoi!)

#### Version Python supportée:
Python 3.6

#### Documentation
Une documentation web est disponible à la racine de l'URL de l'API.


#### Les algorithmes disponibles


| Algorithme | Famille |  Valeur paramètre |    
| ------------- |: -------------: | ---------:|
|**Thompson Sampling**| Bandits |        ts  |
|**Upper Confidence Bound**| Bandits| ucb|
|**LinUCB** | Bandits Contextuels |  linucb|


#### Description des contextes

Lors de la création d'une instance de bandit contextuel, la description du contexte est obligatoire.

Il existe plusieurs type de variables:

###### Les variables réelles

Une variable réelle est un vecteur dense à plusieurs dimensions dont les éléments sont des réels (par exemple [1.,2.]).

*Code json*:

    {"type": "FT_REAL", "dimension": n, "name": "my name 1"}
    


* type: "FT_REAL"
* dimension: la dimension de la variable
* name (facultatif): le nom de la variable


*Valeur de la variable* : Un réel si la dimension est 1, une liste de réels sinon.

###### Les variables catégorielles numériques

Une variable catégorielle numérique est un nombre representant une catégorie.

Par exemple, dans le cas d'une variable representant le sexe d'un utilisateur:

 | Catégorie |  Valeur numérique |
| ------------- |: -------------: |
|Non renseigné| 0 | 
|Homme| 1 |
|Femme| 2 | 
 
 La cardinalité de cette variable est 3.
 
 *Code json*:

    {"type": "FT_CAT_NUMBER", "cardinality": n, "name": "my name 2"}
    
* type: "FT_CAT_NUMBER"
* cardinality: la cardinalité de la variable
* name (facultatif): le nom de la variable

*Valeur de la variable*: Sa valeur numérique.

*A noter*: Pas optimal pour les cardinalités élevées (>100).

 ###### La description de la totalité du contexte
 
 La description de la totalité du contexte est une liste ordonnée contenant les différentes variables au format json.
 
 Par exemple:
 
 *Code json*:

    [{"type": "FT_REAL", "dimension": 2, "name": "my name 1"}, {"type": "FT_REAL", "dimension": 5}, {"type": "FT_CAT_NUMBER", "cardinality": 5, "name": "my name 2"}]
    

###### Le contexte:
 
 Le contexte est une liste ordonnée de dictionnaires ou de valeurs representant chaque variables.
 
 *Code json d'une variable*
 
     {"name": "nom", "value": valeur } ou valeur
     
 * name (facultatif): le nom de la variable
 * value: la valeur de la variable
 
 *Code json compatible avec le contexte décrit ci-dessus*

    [{"name": "my name 1", value": [1.,2.] },[1.,2.,3.,4.,5],3]
 
 


