# Baseball game prediction

# Intro

<p>
This is my final project submission for ML Engineering at SDSU. The goal of the project was to explore a database 
(provided), calculate features, and use those features to predict the home team wins any given baseball game.
</p>

### Requirements

- Docker<br>
- Baseball Database (provided by Julien Pierret)

### How to use

Make sure the baseball.sql file is in the project root directory. Then run <code>docker-compose up</code> from that
directory.

### About the database

- Provided as a single file: <code>baseball.sql</code>
- 1.25 Gb
- 30 unique tables
- Data on more than 16,000 baseball games across 5 seasons (2007-2012)
- Complete game data for > 10,000 of those games
- More than 1 million plate appearances

# Feature Extraction

### Exploring Database

The initial feature set was calculated in SQL (code provided). Features were grouped into 4 categories; Pitching
statistics, Batter statistics, and Team statistics
<br><br>
Each feature was calculated using 3 methods; career average, season average, and 100 day rolling average leading up to
each game.
<br><br>
Season data was not given, so it was calculated based off the game date. I plotted the game dates to confirm no
outlier games extended past the usual end of a season.

![game dates plot](./images/game-dates.png?raw=true)

Seasons look good, so splitting data by season is going to be easy. The dataset originally contained data from some
international league games, but I joined on complete game data and those seem to have dropped out.

### Combining Features

The first step in managing the feature set was to see which pairs of predictors were highly correlated. There is no
point in passing multiple highly similar features to the model. Also, this game me a good look into the features. I
calculated correlation metrics between each pair of continuous features and plotted using plotly.

![pre-combined correlation plot](./images/Predictor-correlations-pre-combine.png?raw=true)

After seeing this plot, I decided that the first step was to merge each home and away feature pair into a single
feature; the difference between the two. I also decided to remove some of the redundant features. For example, at
pitcher bats and plate appearances over the previous 100 games are highly correlated.

![population heatmap at bats v. num games plot](./images/pop-heatmap-num-games-at-bat.png?raw=true)

Pitchers that have played in more games tend to have faced more batters. Throwing both of these features at the model
it probably unnecessary. Instead, I can combine 4 features (home pitcher number of games, away pitcher number of
games, home pitcher number of batters faced, and away pitcher number of batters faced) into a single feature:
difference between number of games played over the last 100 days between the home and away starting pitcher.

To look more closely at this new feature, I can plot bin the feature in a histogram and plot the mean response for
each bin.

![binned mean response-pitcher num games diff plot](./images/MSD-num-games-diff.png?raw=true)

Other than some weird spikes at the low population tails, I can see a clear pattern here. The greater the difference,
the more likely the home team is to win the game. Because the distribution is very normal, I'm not too worried about
the tails throwing the model off considerably.

The next step after merging and pruning was to explore the quality of my remaining features.

![RF importance plot](./images/RF-importances.png?raw=true)

### Cheating features

Some of these features look suspiciously good. Taking a closer look at the best features shows that I'm clearly leaking
something that I shouldn't.

![pitcher stats cheating](./images/pitcher-stats-cheating.png?raw=true)

It looks like my pitcher season stats are cheating. Digging into the code I found that pitching season stats leaked
future data for the current season.

![pitcher stats cheating](./images/streak-cheating.png?raw=true)

Team streak calculations were also done incorrectly. I was accidentally taking the current game into account. Removing
these features dropped the model accuracy from >90% to ~55%.

![RF importance 2 plot](./images/RF-importances-removing-cheaters.png?raw=true)

Without the cheating features, we see most of the remaining features have some importance weight.

Some features are clearly dominating. How many triples of doubles a team has hit or a pitcher has allowed recently
seem to barely matter. After team win rates, the most important features are how good the starting pitchers are
compared to each other (strikeout and WHIP difference), how the teams have been preforming recently (strikeout and OBP).

The most noticeable block of features that is missing from this analysis is the batter statistics. The provided dataset
had the outcomes for each plate appearance. But a big piece of missing information was which batters were in the
starting lineup. The provided lineup table had errors and sometimes included more than 10 players. There was no easy
way for me to pass which players had started in a game when testing the model without leaking information about late
game strategy. If a team was down by a lot in the 9th put a rookie at first base to give their star player a break, my
model was going to see it. Furthermore, team stats always seemed to preform just as well if not better than individual
player stats. The easiest fix for this was to just drop all individual batter statistics and focus on team performance
as a whole.

# Game Predictions

### Train-Test split

Because games occur in a chronological order and team performance changes over time as players are traded or injured,
a traditional 80/20 train/test split doesn't really work. I don't want the model to be trained on any data that happens
after any test data games. I decided to train the model on the 2007-2010 seasons and retain the 2011 season data for
testing.

The downside to this method is that teams change, sometimes substantially, between seasons. I haven't tried it yet,
but the model performance might be better if I set up different models for each season and trained on the first 2/3rds
of each season and tested on the last 1/3rd.

### Models selected

I tried 3 models from SKlearn

- [stochastic gradient descent (SGD)](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html)
- [k-nearest neighbor (KNN)](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
- [random forest classifier (RF)](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)

Each was set up as a pipeline. Data was normalized using a
[Standard Scaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) included in each of the 3 pipelines.

### Results

| **Method**               | **RF** | **KNN** |    **SGD** |
| ------------------------ | -----: | ------: | ---------: |
| Before Removing Cheaters | 90.11% |  89.95% |     89.19% |
| After Removing Cheaters  | 57.99% |  53.27% |     58.60% |
| After Feature Cleanup    | 60.36% |  59.55% | **61.01%** |
