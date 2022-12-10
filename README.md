# Baseball game prediction

# Intro

<p>
This is my final project submission for ML Engineering at SDSU. The goal of the project was to explore a database 
(provided), calculate features, and use those features to predict the outcomes of baseball games.
</p>

### Requirements

- Docker<br>
- Baseball Database (provided by Julien Pierret)

### How to use

Make sure the baseball.sql file is in the project root directory. Then run <code>docker-compose up</code> from that directory.

### About the database

- Provided as a single file: <code>baseball.sql</code>
- 1.25 Gb
- 30 unique tables
- Data on more than 16,000 baseball games across 5 seasons (2007-2012)
- Complete game data for > 10,000 of those games
- More than 1 million plate appearances

# Feature Extraction

### Exploring Database

The initial feature set was calculated in SQL (code provided). Features were grouped into 4 categories; Pitching statistics, Batter statistics, and Team statistics
<br><br>
Each feature was calculated using 3 methods; career average, season average, and 100 day rolling average leading up to each game.
<br><br>
Season data was not given, so it was calculated based off the game date. I plotted the game dates to confirm no outlier games extended past the usual end of a season.
<br>
![alt text](https://github.com/seanfahey1/MLEngineering/blob/final/images/game-dates.html?raw=true)
<br><br>

### Feature pruning

### Combining features

# Game Predictions

### Train-Test split

### Models selected

### Best model

###
