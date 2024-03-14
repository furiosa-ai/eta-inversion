for t in inv edit metrics data # eval
do
mkdir -p result/test
python test/test_${t}.py 2>&1 | tee result/test/test_${t}_log.txt
done