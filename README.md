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

Some features are clearly dominating. How many triples of doubles a team has hit or a pitcher has allowed recently
seem to barely matter. The most important features are how good the starting pitchers are compared to each other
(season runs and hits difference), how the teams have been preforming recently (current streak), and how many games
each team has won over the season.

# Game Predictions

### Train-Test split

### Models selected

### Best model

###
