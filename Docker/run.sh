sleep 10

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

mysql --user=root --password=fahey -h db < 100_day_rolling_calc.sql > output/results.txt

date >> output/log.txt
echo "" >> output/log.txt
