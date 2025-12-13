# League of Legends Analysis: Wards vs. Dragons

## Introduction

### General Introduction

League of Legends is a popular multiplayer game, in which two teams of five players compete against one another, attempting to gain an advantage over the opposing team, and ultimately, reaching and destroying their base. Outside of standard play, there is a large competetive scene, where preorganized groups compete in organized tournaments. Of competetive play, the largest and most popular of such would be Worlds, a yearly competition in which select, qualified teams from different regions around the world come together and compete for the chance to win the world title.  

There are various ways that a team could get ahead in a game, with one of such methods being through slaying dragons. This objective (that of slaying the dragon/ dragons), could be vital to the state of the game, as upon completing such objectives, the team is granted benefits, such as increased movement speed, reduced skill cooldowns, and more. 

Another key aspect of strategy in League of Legends is vision. Teams place "wards" on the map, effectively lighting up previously hidden areas of the map, helping teams track the positions of enemies.

This increased visibility can be especially important when contesting dragons, as it provides critical information on the opposing team’s position and intentions, enabling better coordination and decision-making around these high-value objectives.

Our central research question is:
How strongly is a team’s vision control, measured by the number of wards they place (and wards per minute), associated with the number of dragons they secure in a Worlds game?

This question matters both to players and to analysts. If we find a strong relationship, it would support the idea that investing in vision is a key part of objective control strategy at the highest level of play. On the other hand, if the relationship is weak, it could suggest that other factors (like early‑game lane pressure, jungle pathing, or team‑fight execution) are more important than raw ward count when it comes to securing dragons. Throughout the rest of this project, we’ll explore this relationship through exploratory data analysis, hypothesis testing, evaluating missingness, and predictive modeling.

### Introduction of Columns

For the rest of this project, we focus on a few key columns that are most relevant to our question:

gameid: A unique identifier for each match, used to group the two teams in the same game.

league: The competetive region in which the game is taking place.

teamname: The name or identifier of the team.

gamelength: The duration of the game (in seconds), used to normalize columns

wpm: A derived feature equal to wards_placed divided by game length in minutes, which measures how actively a team uses vision over time.

wcpm: The number of enemy wards the team destroyed, divided by game length. This captures how effectively the team denies the enemy’s vision. 

controlwardsbought: The total number of control wards the team purchased. Control wards are special wards that reveal and disable enemy wards, so higher values indicate a stronger focus on long-term, persistent vision. 

visionscore: Riot’s (Creator of League of Legends) composite vision metric that rewards placing useful wards, clearing enemy wards, and providing vision of unseen enemies and objectives. A higher vision score reflects better overall vision control, not just raw ward counts. 

dragons: The number of elemental dragons a team secured in that game.

cwpm: Number of control wards placed per minute, normalized using gamelength

dragons_per_minute: Number of dragons secured, normalized using gamelength

supvis: Vision score given by Riot for only the support. Useful as supports typically are tasked with the role of gaining and controling vision (though other teammates may help)

supcont: Number of control wards placed by the support

## Data Cleaning and Exploratory Data Analysis

The columns that we kept for the purpose of analysis consisted of:

```python
['gamelength', 'league', 'wpm', 'wcpm', 'controlwardsbought', 'visionscore',
 'dragons', 'cwpm', 'dragons_per_minute', 'supvis', 'supcont']
```

Of the 12 years of Worlds data collected, we decided to exclude years 2014-2018 from our analysis, as during that time frame, control wards were not a part of the game. And only after 2018, control wards have not been updated.

We Concatenated the remaining 7 years into a two seperate DataFrames, being individual_data_df and team_data_df. From the total 883488 rows, we extracted the support related data (support vision score and support control wards) and added them as columns to the team data, we were left with a sibgle DataFrame, indexed by gameid and teamname, of 147186 rows. 

From the data, we had missingness in columns, wpm, wcpm, cwpm, supvis and supcount, which we imputed using the median of their respective columns.

# add dataframe.head here

### Univariate Analysis 

<iframe src="histogram_game_length.html" width=800 height=620 frameBorder=0></iframe>

From this graph, we can see that the average length of a game follows a normal distribution, but is slightly right skewed. 

### Bivariate Analysis

<iframe src="distributionWPMDC.html" width=800 height=620 frameBorder=0></iframe>

This histogram shows that as game length increases, so does the number of dragons secured

## Assessment of Missingness



### MAR Analysis

goldat25 is missing for about 12% of team rows. Conceptually, this missingness likely happens when the game’s statistics at 25 minutes are not recorded or are undefined (for example, games that end very early, or games in competitions where Oracle’s Elixir did not track 25‑minute stats). In other words, the probability that goldat25 is missing seems to depend on other recorded variables, such as the game’s gamelength or the league, rather than on the (unobserved) value of goldat25 itself.

Because the missingness in goldat25 can be explained using observed variables in the dataset, it is more consistent with Missing At Random (MAR) than Not Missing At Random (NMAR). To make the MAR assumption even more plausible, additional metadata about data collection (e.g. which leagues/patches have 25‑minute stats available) would be helpful; if we had such information, we could model missingness using those observed columns directly.

Null hypothesis:
Missingness of goldat25 is independent of game length. The average gamelength_min is the same for rows where goldat25 is missing and where it is observed.

Alternative:
Missingness of goldat25 depends on game length. The average gamelength_min is different between the two groups.

# graph goes here

As our observed value of -10 is far out from the null distribution, therefore, having a p_value of 0, we can reject the null hypothesis. 

## Hypothesis Testing

## Framing a Prediction Problem

## Baseline Model

In our baseline model, we used a Random Forest Classifier with the features: ```wpm```, ```wcpm```, ```cwpm```. While all three of these features were quantitative, only two of them were provided in the original Data Tables. As mentioned earlier, we standardized ```controlwardsplaced```, dividing by the length of the game (in minutes), creating the new column ```cwpm```.

After fitting a baseline model, our model resulted in a Training Accuracy of ```0.9992```. Out model, however, had a Test Accuracy of ```0.5469```, much lower than the Training Accuracy, suggesting that our model was likely over fitting the data.

## Step 7: Final Model

In our final model, we added three new features to the data: ```visionscore```, ```supvis```, and ```supcont```. We chose to add these features to our model because in the game of Leauge of Legends, the supports typically have the largest impact on vision score, having dedicated items allowing for them to place more wards and clear wards more easily. Additionally, supports are often tasked with setting up for objectives, which is often associated with the use of control wards.

## Fairness Analysis
We plan to check the fairness of our model through a comparison of two groups. The question that we decided to test was: Does our model perform worse for players from the Korean Region (LCK) relative to those of the North American Region (NLC)? 

To assess this, we ran a permutation test, evaluating the diffence in accuracies between the two groups, with a p-value cutoff of 0.05.


Null hypothesis: The model is fair. The accuracy calculated for players from the Korean Region (LCK) is the same as the accuracy calculated from those in the North American Region (NLC).

Null hypothesis: The model is unfair. The accuracy calculated for players from the Korean Region (LCK) is not the same as the accuracy calculated from those in the North American Region (NLC).



















