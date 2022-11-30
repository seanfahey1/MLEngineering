if ! mysql --user=root --password=fahey21 -h db -e 'use baseballdb'; then
    echo 'BaseballDB not found'
    mysqladmin --user=root --password=fahey21 -P 3306 --protocol=tcp create baseballdb
    echo 'Created BaseballDB'
    mysql --user=root --password=fahey21 -h db --database=baseballdb < baseball.sql
    echo 'Successfully updated BaseballDB'
else
  echo 'BaseballDB found'
fi

mysql --user=root --password=fahey21 -h db -e '
  use baseballdb;

  create table if not exists test_table
  select
    position
  from position;' > /app/output/test.txt

echo 'done!'
