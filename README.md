## API Décisions: Ok Corral

Api pour la prise de décision en ligne à information partielle (pour faire du bandit quoi!)

### Version Python supportée:
Python 3.6

### Documentation
Une documentation web est disponible à la racine de l'URL de l'API.


### Les algorithmes disponibles


 Algorithme | Famille |  Valeur paramètre    
 --- | --- | --- |
**Thompson Sampling**| Bandits |        ts  
**Upper Confidence Bound**| Bandits| ucb
**LinUCB** | Bandits Contextuels |  linucb


### Description des contextes

Lors de la création d'une instance de bandit contextuel, la description du contexte est obligatoire.

Il existe plusieurs type de variables:

##### Les variables réelles

Une variable réelle est un vecteur dense à plusieurs dimensions dont les éléments sont des réels (par exemple [1.,2.]).

*Code json*:

    {"type": "FT_REAL", "dimension": n, "name": "my name 1"}
    


* type: "FT_REAL"
* dimension: la dimension de la variable
* name: le nom de la variable


*Valeur de la variable* : Un réel si la dimension est 1, une liste de réels sinon.

##### Les variables catégorielles numériques

Une variable catégorielle numérique est un nombre representant une catégorie.

Par exemple, dans le cas d'une variable representant le sexe d'un utilisateur:

 Catégorie   |  Valeur numérique 
 ---  | --- 
Non renseigné| 0 
Homme| 1 
Femme| 2 
 
 La cardinalité de cette variable est 3.
 
 *Code json*:

    {"type": "FT_CAT_NUMBER", "cardinality": n, "name": "my name 2"}
    
* type: "FT_CAT_NUMBER"
* cardinality: la cardinalité de la variable
* name (facultatif): le nom de la variable

*Valeur de la variable*: Sa valeur numérique.

*A noter*: Pas optimal pour les cardinalités élevées (>100).

##### La description de la totalité du contexte
 
 La description de la totalité du contexte est une liste contenant les différentes variables au format json.
 
 Par exemple:
 
 *Code json*:

    [{"type": "FT_REAL", "dimension": 2, "name": "my name 1"}, {"type": "FT_REAL", "dimension": 5, name: "myname2"}, {"type": "FT_CAT_NUMBER", "cardinality": 5, "name": "my name 3"}]
    
 
 Si les variables contextuelles diffèrent suivant les actions, alors fournir une liste ordonnée des différentes descriptions:
 
 *Code json*:
 
    [[{"type": "FT_REAL", "dimension": 2, "name": "my name 1"}], [{"type": "FT_REAL", "dimension": 2, "name": "my name 1"}]]

##### Le contexte:

 Le contexte est un dictionnaire de valeurs indexés par leur noms.
 
 *Code json d'une variable*
 
     "name": valeur
     
 * name: le nom de la variable
 * value: la valeur de la variable
 
 *Code json compatible avec le contexte décrit ci-dessus*

    {"my name 1": [1.,2.], "myname2": [1.,2.,3.,4.,5], "my name 3": 3}
 
 Si les variables contextuelles diffèrent suivant les actions, alors la syntaxe des contextes est:

     {"shared_context": contexte, "contexts": [[action, contexte],...[action,contexte]] }

- **L'id retourné suite à la prise de décision est l'id du couple action/contexte et non l'id de l'action. La mise à jour doit cependant toujours être effectuée avec un seul contexte et l'id de l'action.**
- "shared_context" est un dictionnaire contenant les variables en commun avec toutes les classes.
- "contexts" est une liste de couples contenant l'id de l'action ansi que les variables de contextes spécifiques.
- Les dictionnaires réprésentant les variables peuvent être vite tant qu'il y a une variable par action dans "shared_context" ou "contexts".
- Potentiellement, toutes les variables peuvent être mises dans "shared_context" et uniquement les variables précédements déclarées.
seront prises en compte par les différentes actions.
- S'il n'y a pas de "shared_context", on peut directement donner la liste des couples action/contexte.
- Placer une variable dans "contexts" permet de donner des valeurs différentes à une variable suivant les actions.
- Une action peut avoir plusieurs occurences dans "contexts" avec des valeurs de variables différentes.
- Il ne doit pas y avoir de conflits de nom de variables entre "shared_context" et "contexts". En cas de conflit, la valeur de la variable déclarée dans "contexts" écrasera la valeur de "shared_context". Ceci est un effet de bord, ne fait pas partie du contrat d'interface et ce comportement peut changer dans les versions futures de l'API.
