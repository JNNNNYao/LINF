echo 'set5' &&
echo 'x2' &&
python test.py --config ./configs/test/test-fast-set5-2.yaml --model $1 --gpu $2 --sample 0 --detail --temperature $3 --randomness --window 8 &&
echo 'x3' &&
python test.py --config ./configs/test/test-fast-set5-3.yaml --model $1 --gpu $2 --sample 0 --detail --temperature $3 --randomness --window 8 &&
echo 'x4' &&
python test.py --config ./configs/test/test-fast-set5-4.yaml --model $1 --gpu $2 --sample 0 --detail --temperature $3 --randomness --window 8 &&
echo 'x6*' &&
python test.py --config ./configs/test/test-fast-set5-6.yaml --model $1 --gpu $2 --sample 0 --detail --temperature $3 --randomness --window 8 &&
echo 'x8*' &&
python test.py --config ./configs/test/test-fast-set5-8.yaml --model $1 --gpu $2 --sample 0 --detail --temperature $3 --randomness --window 8 &&

echo 'set14' &&
echo 'x2' &&
python test.py --config ./configs/test/test-fast-set14-2.yaml --model $1 --gpu $2 --sample 0 --detail --temperature $3 --randomness --window 8 &&
echo 'x3' &&
python test.py --config ./configs/test/test-fast-set14-3.yaml --model $1 --gpu $2 --sample 0 --detail --temperature $3 --randomness --window 8 &&
echo 'x4' &&
python test.py --config ./configs/test/test-fast-set14-4.yaml --model $1 --gpu $2 --sample 0 --detail --temperature $3 --randomness --window 8 &&
echo 'x6*' &&
python test.py --config ./configs/test/test-fast-set14-6.yaml --model $1 --gpu $2 --sample 0 --detail --temperature $3 --randomness --window 8 &&
echo 'x8*' &&
python test.py --config ./configs/test/test-fast-set14-8.yaml --model $1 --gpu $2 --sample 0 --detail --temperature $3 --randomness --window 8 &&

echo 'b100' &&
echo 'x2' &&
python test.py --config ./configs/test/test-fast-b100-2.yaml --model $1 --gpu $2 --sample 0 --detail --temperature $3 --randomness --window 8 &&
echo 'x3' &&
python test.py --config ./configs/test/test-fast-b100-3.yaml --model $1 --gpu $2 --sample 0 --detail --temperature $3 --randomness --window 8 &&
echo 'x4' &&
python test.py --config ./configs/test/test-fast-b100-4.yaml --model $1 --gpu $2 --sample 0 --detail --temperature $3 --randomness --window 8 &&
echo 'x6*' &&
python test.py --config ./configs/test/test-fast-b100-6.yaml --model $1 --gpu $2 --sample 0 --detail --temperature $3 --randomness --window 8 &&
echo 'x8*' &&
python test.py --config ./configs/test/test-fast-b100-8.yaml --model $1 --gpu $2 --sample 0 --detail --temperature $3 --randomness --window 8 &&

echo 'urban100' &&
echo 'x2' &&
python test.py --config ./configs/test/test-fast-urban100-2.yaml --model $1 --gpu $2 --sample 0 --detail --temperature $3 --randomness --window 8 &&
echo 'x3' &&
python test.py --config ./configs/test/test-fast-urban100-3.yaml --model $1 --gpu $2 --sample 0 --detail --temperature $3 --randomness --window 8 &&
echo 'x4' &&
python test.py --config ./configs/test/test-fast-urban100-4.yaml --model $1 --gpu $2 --sample 0 --detail --temperature $3 --randomness --window 8 &&
echo 'x6*' &&
python test.py --config ./configs/test/test-fast-urban100-6.yaml --model $1 --gpu $2 --sample 0 --detail --temperature $3 --randomness --window 8 &&
echo 'x8*' &&
python test.py --config ./configs/test/test-fast-urban100-8.yaml --model $1 --gpu $2 --sample 0 --detail --temperature $3 --randomness --window 8 &&

true
