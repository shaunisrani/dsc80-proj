# League of Legends Analysis: Wards vs. Dragons

Authors: Shaun Israni, Jared Wang

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

We Concatenated the remaining 7 years into a two seperate DataFrames, being individual_data_df and team_data_df. From the total 883488 rows, we extracted the support related data (support vision score and support control wards) and added them as columns to the team data, we were left with a sibgle DataFrame, indexed by gameid and teamname, of 147186 rows. From the data, we had missingness in columns, wpm, wcpm, cwpm, supvis and supcount, which we imputed using the median of their respective columns.

Below is the head of our cleaned DataFrame:

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>league</th>
      <th>gamelength</th>
      <th>wpm</th>
      <th>wcpm</th>
      <th>...</th>
      <th>cwpm</th>
      <th>dragons_per_minute</th>
      <th>supvis</th>
      <th>supcont</th>
    </tr>
    <tr>
      <th>gameid</th>
      <th>teamname</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">ESPORTSTMNT01/1030526</th>
      <th>Flamengo MDL</th>
      <td>CBLOL</td>
      <td>1770</td>
      <td>2.88</td>
      <td>1.46</td>
      <td>...</td>
      <td>1.02</td>
      <td>1.69e-03</td>
      <td>78.0</td>
      <td>12.0</td>
    </tr>
    <tr>
      <th>KaBuM! Esports</th>
      <td>CBLOL</td>
      <td>1770</td>
      <td>2.98</td>
      <td>1.12</td>
      <td>...</td>
      <td>1.46</td>
      <td>0.00e+00</td>
      <td>58.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">ESPORTSTMNT01/1040501</th>
      <th>Vivo Keyd</th>
      <td>CBLOL</td>
      <td>2362</td>
      <td>3.76</td>
      <td>0.99</td>
      <td>...</td>
      <td>1.50</td>
      <td>0.00e+00</td>
      <td>106.0</td>
      <td>25.0</td>
    </tr>
    <tr>
      <th>Uppercut esports</th>
      <td>CBLOL</td>
      <td>2362</td>
      <td>3.40</td>
      <td>1.40</td>
      <td>...</td>
      <td>1.32</td>
      <td>2.12e-03</td>
      <td>99.0</td>
      <td>18.0</td>
    </tr>
    <tr>
      <th>ESPORTSTMNT01/1040511</th>
      <th>ProGaming Esports</th>
      <td>CBLOL</td>
      <td>2128</td>
      <td>3.27</td>
      <td>1.18</td>
      <td>...</td>
      <td>1.30</td>
      <td>0.00e+00</td>
      <td>82.0</td>
      <td>14.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 11 columns</p>
</div>

### Univariate Analysis 

This is a graph, depicting the distribution of game length.

<iframe src="histogram_game_length.html" width=800 height=620 frameBorder=0></iframe>

From this graph, we can see that the average length of a game follows a normal distribution, but is slightly right skewed. 

### Bivariate Analysis

Below is a bivariative graph displaying the relationship between game length and dragons secured. 

<iframe src="distributionWPMDC.html" width=800 height=620 frameBorder=0></iframe>

This histogram shows that an increasing trend, where as game length increased, the number of dragons followed suit.

### Interesting Aggregates

Below is an aggregated table from our dataset.

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>wpm_quartile</th>
      <th>mean_wpm</th>
      <th>mean_dragons</th>
      <th>mean_dragons_per_min</th>
      <th>mean_visionscore</th>
      <th>n_games</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Low</td>
      <td>2.47</td>
      <td>1.82</td>
      <td>1.04e-03</td>
      <td>179.49</td>
      <td>36391</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Med-Low</td>
      <td>2.94</td>
      <td>2.18</td>
      <td>1.16e-03</td>
      <td>222.54</td>
      <td>36426</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Med-High</td>
      <td>3.28</td>
      <td>2.35</td>
      <td>1.20e-03</td>
      <td>251.08</td>
      <td>36353</td>
    </tr>
    <tr>
      <th>3</th>
      <td>High</td>
      <td>3.87</td>
      <td>2.55</td>
      <td>1.25e-03</td>
      <td>292.85</td>
      <td>36380</td>
    </tr>
  </tbody>
</table>

We grouped teams into four “vision tiers” based on their wards per minute (WPM) using quartiles. The resulting table shows that as we move from the Low WPM group to the High WPM group, the average number of dragons secured per game and dragons per minute both increase. The high‑vision groups also tend to have higher average vision scores. This aggregated view supports the idea that teams that invest more heavily in vision tend to control more dragons over the course of a game.

## Assessment of Missingness

### NMAR Analysis: 

In our cleaned team‐level dataset, the columns we actually use for modeling (`wardsplaced`, `wpm`, `wcpm`, `controlwardsbought`, `visionscore`, `totalgold`, `dragons`, and `dragons_per_min`) have no missing values for team rows. To study missingness, we therefore focused on the team‐level column **`goldat25`**, which records the total gold a team has at the 25‑minute mark.

`goldat25` is missing for 682 out of 18,292 team rows (about 3.7%). Conceptually, this missingness likely happens when the game’s 25‑minute statistics are not defined or not recorded – for example, in games that end early or in contexts where Oracle’s Elixir does not track 25‑minute snapshots. In other words, the probability that `goldat25` is missing seems to depend on **other recorded variables**, such as the game’s length, rather than on the (unobserved) value of `goldat25` itself.

Because the missingness in `goldat25` can be explained using observed variables in the dataset, it is more consistent with Missing At Random (MAR) than Not Missing At Random (NMAR). I do not believe `goldat25` is NMAR. To make the MAR assumption even more plausible, additional metadata about data collection (ex: which leagues or patches have 25‑minute stats available, or flags for games that ended before 25 minutes) would be helpful; if we had this information, we could model missingness directly using those observed variables.


### Missingness Dependency:
Column chosen for missingness analysis:
We analyzed the missingness of the team‑level column goldat25, which records the total gold a team has at the 25‑minute mark. In our 2025 Oracle’s Elixir team‑level data, goldat25 is missing for 2,316 out of 19,926 team rows (~11.6%), making it a good candidate for assessing missingness patterns. The main columns we use for our main analysis (wardsplaced, wpm, visionscore, dragons, etc.) have no missing values for team rows.


Does missingness of goldat25 depend on game length?

Our first hypothesis was that the missingness of goldat25 depends on the length of the game. Intuitively, statistics at 25 minutes might not be recorded in some games, particularly shorter ones.

Null hypothesis (H₀): Missingness of goldat25 is independent of gamelength_min. The average game length is the same for rows where goldat25 is missing and where it is observed.

Alternative hypothesis (H₁): Missingness of goldat25 depends on gamelength_min. The average game length differs between missing and non‑missing rows.

We created an indicator variable gold25_missing (1 if goldat25 is missing, 0 otherwise) and used the difference in mean gamelength_min between the missing and non‑missing groups as our test statistic. The observed difference was about −2.9 minutes, meaning that games with missing goldat25 are, on average, roughly 2.9 minutes shorter than games with recorded goldat25.

To assess significance, we performed a permutation test with 2,000 shuffles: we repeatedly permuted the gold25_missing labels across teams, recomputed the difference in mean gamelength_min, and built a null distribution. The empirical p‑value (two‑sided) was < 0.001 – none of the permuted differences were as extreme as the observed difference.
A histogram of the null distribution shows a tight bell‑shaped curve centered near 0, with our observed statistic far out in the tail.

Conclusion: We reject the null hypothesis and conclude that the missingness of goldat25 does depend on game length. Games with missing goldat25 tend to be shorter on average.

<iframe src="gold_missing_gamelength.html" width=800 height=620 frameBorder=0></iframe>

Blue bars (or whatever color Plotly picks for False) = distribution of game lengths when goldat25 is not missing.
Orange bars (True) = distribution when goldat25 is missing.
You should see the “missing” distribution shifted left (shorter games), matching the negative difference in means and tiny p‑value.


Does missingness of goldat25 depend on side (Blue vs Red)?

We also tested whether missingness of goldat25 depends on which side the team was on in the game (Blue or Red).

Null hypothesis (H₀): Missingness of goldat25 is independent of side; the proportion of Blue teams is the same in the missing and non‑missing groups.

Alternative hypothesis (H₁): Missingness of goldat25 depends on side; the proportion of Blue teams differs between groups.

We encoded side as a binary variable (side_blue = 1 for Blue, 0 for Red) and again used the difference in group means (equivalently, the difference in the proportion of Blue teams) as our test statistic. The observed difference was essentially 0.0, and a permutation test with 2,000 shuffles yielded a p‑value of approximately 1.0. The cross‑tabulation of gold25_missing and side shows exactly equal counts of Blue and Red teams in both the missing and non‑missing groups.

Conclusion: We fail to reject the null hypothesis and conclude that the missingness of goldat25 does not depend on the team’s side. This makes sense, because goldat25 is recorded (or not) at the game level, and both teams share the same game.

<iframe src="gold_missing_null_diff_glen" width=800 height=620 frameBorder=0></iframe>

The blue histogram is what differences in mean game length look like under the null (when we randomly shuffle the missingness labels).
The red vertical line is the observed difference (~−10 minutes), far out in the tail → strong evidence of dependence


Overall, our analysis suggests that the missingness in goldat25 is not MCAR: it strongly depends on game‑level characteristics like gamelength_min, but not on irrelevant features like side. Since this dependence can be explained using variables in our dataset, we treat the missingness in goldat25 as Missing At Random (MAR) rather than NMAR.

## Hypothesis Testing
Hypothesis Test: Do higher warding rates lead to more dragons?

Our main research question is whether teams that invest more in vision control secure more dragons. To formalize this, we used the cleaned team‑level dataset from Step 2 and focused on:

-wpm: wards placed per minute
-dragons: number of elemental dragons secured by a team in a game

We split teams into two groups based on the median value of wpm:

-High‑WPM group: teams with wpm ≥ median
-Low‑WPM group: teams with wpm < median

Our goal was to test whether high‑WPM teams tend to secure more dragons than low‑WPM teams.

Null hypothesis (H₀):
The average number of dragons secured is the same for high‑WPM and low‑WPM teams. Any observed difference in mean dragons is due to random variation.

Alternative hypothesis (H₁):
High‑WPM teams secure more dragons on average than low‑WPM teams. (One‑sided: mean_dragons_high > mean_dragons_low.)

We used the difference in group means: diff = dragons_high WPM − dragons_low WPM, as our test statistic. This is a natural choice because it directly measures how much more objective control high‑vision teams have, on average.

In the observed data, high‑WPM teams secured about 2.44 dragons per game, while low‑WPM teams secured about 2.00 dragons per game, for an observed difference of: diff ≈ 0.44 dragons.

To assess whether this difference could plausibly arise by chance under the null, we performed a permutation test:

1. We repeatedly (5,000 times) shuffled the high_wpm labels among teams, breaking any real association between warding and dragons while keeping the overall distribution of dragons the same.

2. For each shuffle, we recomputed the difference in mean dragons between the “high” and “low” groups, building a null distribution of the test statistic.

3. The one‑sided p‑value was estimated as the proportion of shuffled differences that were at least as large as the observed difference.

In our simulation, none of the permuted differences were as large as the observed difference of ~0.44 dragons. With 5,000 permutations, this yields a one‑sided p‑value of p < 0.0002 (and a two‑sided p‑value < 0.0004); A histogram of the permutation distribution shows a tight bell‑shaped curve centered near 0, while the observed statistic appears far in the right tail.

<iframe src="plotc.html" width=800 height=620 frameBorder=0></iframe>

Conclusion:
At the 5% significance level, we reject the null hypothesis and find strong evidence that teams with higher warding rates (higher wpm) secure more dragons on average than teams with lower warding rates. While this test does not prove a causal relationship, it supports the idea that investing in vision is associated with better control of dragon objectives at the professional level.

## Framing a Prediction Problem

Since we saw a positive coorelation between the wards per minute and the number of dragons secured, we wanted to see if there was any relationship between other vision related metrics. More specifically, we wanted to ask, can we predict if a team ended up securing more dragons than their opponent based solely on vision related data?

Our prediction model was focused on teamwide data, however, with an emphasis on the support role. The model that we build for this falls under Classification, where we Biranized the dragons column, assigning the value 1 if a team secured more dragons than their opponents, and 0 otherwise within a game. To address ties, we assigned the value 0 to both teams, as neither team has "more" dragons than the other. 

We split our data into two parts, consisting of 70% training, and 30% test data. To evaluate this model, we considered accuracy as our primary metric, as our data was relatively balanced, skewed slightly by out assigning of 0 to ties. In our DataFrame, we had a Brianized ratio of 58% 0's, and 44% 1's.

## Baseline Model

In our baseline model, we used a Random Forest Classifier with the features: ```wpm```, ```wcpm```, ```cwpm```. While all three of these features were quantitative, only two of them were provided in the original Data Tables. As mentioned earlier, we standardized ```controlwardsplaced```, dividing by the length of the game (in minutes), creating the new column ```cwpm```.

After fitting a baseline model, our model resulted in a Training Accuracy of ```0.9992```. Out model, however, had a Test Accuracy of ```0.5469```, much lower than the Training Accuracy, suggesting that our model was likely over fitting the data. Based on this low relative test accuracy, we concluded that there was likely room for improvement in this model.

## Final Model

In our final model, we added three new features to the data: ```visionscore```, ```supvis```, and ```supcont```. We chose to vision score to our features as it accounted for how long wards stayed alive, giving an additional metric to the number of wards placed. Additionally, we wanted to include support related data as in the game of Leauge of Legends, the supports typically have the largest impact on vision score. They often purchase dedicated items allowing for them to place more wards and clear opposing wards more easily, and are often tasked with setting up for objectives, which is often associated with the use of control wards.

Our Final Model, similar to that of our base, was also a Random Forest Classifier. We captured the support related data through a query on position, and added these columns to our team based data. To fine tune our model, we tested different parameters, consisting of: ```max_depth```, ```min_samples_split```, ```min_samples_leaf```, and ```n_estimators```. We chose ```max_depth``` as we did not want our model to hyper focus on specific patterns, possibly reducing over fitting. As for ```min_sample_leaf```, we wanted to try and boost our test accuracy, trying to make more data driven predictions. For ```min_samples_split```, we wanted to prevent our model from branching off too much, again, preventing overfitting. As for ```n_estimators```, we wanted to ensure that there were sufficient trees in our model.

With that in mind, we defined our paramater grid as such:

```python
param_grid = {
    "max_depth": [5, 7, 9, None],
    "min_samples_split": [2, 30, 45],
    "min_samples_leaf": [2, 4, 8],
    "n_estimators": [200, 500],
}
```
From our GridSearch, we got the following parameters:

```python
Fitting 3 folds for each of 72 candidates, totalling 216 fits
Best params: {'max_depth': 9, 'min_samples_leaf': 2, 'min_samples_split': 4, 'n_estimators': 500}
```

With our new model, our Training and Validation scores were as such:

```python
Train score: 0.6279530843598174
Validation score: 0.608404900950418
```
While our model accuracy remains quite low, it improvevd from the baseline model, as well as the naive model. From this, we may conclude that the added parameters are not strong parameters for prediction on whether a team secured more dragons or not.

## Fairness Analysis
We plan to check the fairness of our model through a comparison of two groups. The question that we decided to test was: Does our model perform worse for players from the Korean Region (LCK) relative to those of the North American Region (NLC)? 

To assess this, we ran a permutation test, evaluating the diffence in accuracies between the two groups, with a p-value cutoff of 0.05.

With this in mind, we set defined our hypothesis as such:

Null hypothesis: The model is fair. The accuracy calculated for players from the Korean Region (LCK) is the same as the accuracy calculated from those in the North American Region (NLC).

Alternate hypothesis: The model is unfair. The accuracy calculated for players from the Korean Region (LCK) is not the same as the accuracy calculated from those in the North American Region (NLC).

Again, our test statistic was accuracy, however, this time, focusing on the difference between groups. 

Our result showed a p_value of 0.1728, which is higher than our cutoff of 0.05. From this, we conclude that fail to reject the null hypothesis.


















