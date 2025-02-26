echo "etth1"
./scripts/multivariate_forecasting/ETT/iMamba_ETTh1.sh

echo "etth2"
./scripts/multivariate_forecasting/ETT/iMamba_ETTh2.sh

echo "ettm1"
./scripts/multivariate_forecasting/ETT/iMamba_ETTm1.sh

echo "ettm2"
./scripts/multivariate_forecasting/ETT/iMamba_ETTm2.sh


echo "weather start"
./scripts/multivariate_forecasting/Weather/iMamba.sh


echo "electricity start"
./scripts/multivariate_forecasting/ECL/iMamba.sh

echo "traffic start"
./scripts/multivariate_forecasting/Traffic/iMamba.sh

# echo "exchange start"
# ./scripts/multivariate_forecasting/Exchange/iMamba.sh

