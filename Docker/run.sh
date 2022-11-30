
if ! mysql --user=root --password=fahey -h db -e 'use baseballdb'; then
    echo 'BaseballDB not found'
    mysqladmin --user=root --password=fahey -h db create baseballdb
    echo 'Created empty BaseballDB'
    mysql --user=root --password=fahey -h db --database=baseballdb < baseball.sql
    echo 'Successfully updated BaseballDB'
else
  echo 'BaseballDB found'
fi

mysql --user=root --password=fahey -h db < 100_day_rolling_calc.sql > output/test2.txt

echo 'done!'
