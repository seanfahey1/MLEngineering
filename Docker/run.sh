# prevents mysql commands from running before DB setup is finished
sleep 10

# log
date >> output/log.txt

# check if baseballdb exists and create it if it doesn't exist
DB=`mysqlshow --user=root --password=fahey -h db| grep -v Wildcard | grep -o baseballdb`
if [ "$DB" == "baseballdb" ]; then
    echo 'BaseballDB found'
    echo 'BaseballDB found:' >> output/log.txt
else
    echo 'BaseballDB not found'
    echo 'BaseballDB not found:' >> output/log.txt
    mysqladmin --user=root --password=fahey -h db create baseballdb
    echo 'Created empty BaseballDB'
    mysql --user=root --password=fahey -h db --database=baseballdb < baseball.sql
    echo 'Successfully updated BaseballDB'
fi

# check if final table exists and run sql code if it doesn't exist
FEATURESEXIST=`mysqlshow --user=root --password=fahey -h db baseballdb | grep -v Wildcard | grep -o game_features_3`
#if [ "$FEATURESEXIST" != "game_features_3" ]; then
    echo 'Calculating features, this may take a while'
#    mysql --user=root --password=fahey -h db < feature-extract.sql
#else
    echo 'Found feature calculation table'
#fi

# log
date >> output/log.txt
echo "" >> output/log.txt

# run python code
python3 connection-test.py
