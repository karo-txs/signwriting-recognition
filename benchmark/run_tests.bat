@echo off
setlocal enabledelayedexpansion
REM Configuração dos parâmetros
set "OUTPUT_CSV=inference_results.csv"
set "N_INFERENCES=1000"
set "REPEAT=30"  REM Número de repetições para cada configuração

REM Construir a imagem Docker
docker build -t tf_inference_test .

REM Inicializar o arquivo CSV no diretório local (adiciona o cabeçalho apenas uma vez)
if not exist %OUTPUT_CSV% (
    echo ModelPath,Memory,CPU,AvgInferenceTime,StdDev,CI95,Throughput,AvgCPU,AvgMemory > %OUTPUT_CSV%
)

REM Lista de configurações de memória e CPUs 4 2 1 0.5
set memories=8g 4g 2g 1g 512m
set cpus=20 16 8

REM Diretório de trabalho local
set "CURRENT_DIR=%cd%"
echo %CURRENT_DIR%

REM Loop para testar diferentes configurações de memória e CPUs
for %%m in (%memories%) do (
    for %%c in (%cpus%) do (
        echo -----------------------------------------
        echo Rodando teste com: modelo=/app/sw_model.tflite, memória=%%m, CPU=%%c
        echo -----------------------------------------

        REM Remover qualquer container anterior com o mesmo nome
        docker rm -f tf_test_container >nul 2>&1

        REM Iniciar o container em segundo plano com o volume montado
        docker run -d --name tf_test_container -v "%CURRENT_DIR%:/app" --memory="%%m" --cpus="%%c" tf_inference_test tail -f /dev/null

        REM Executar o script Python dentro do container e capturar a saída
        for /f "tokens=1-4 delims=," %%a in ('docker exec tf_test_container python3 /app/inference_test.py --model_path "/app/sw_model.tflite" --n_inferences %N_INFERENCES% --repeat %REPEAT%') do (
            set mean=%%a
            set stddev=%%b
            set ci=%%c
            set tput=%%d
        )

        REM Capturar uso de CPU e Memória com docker stats, mantendo apenas a parte numérica da memória
        for /f "tokens=1,2 delims=, " %%x in ('docker stats --no-stream --format "{{.CPUPerc}},{{.MemUsage}}" tf_test_container') do (
            set cpuusage=%%x
            for /f "tokens=1 delims=M " %%y in ("%%y") do (
                set memusage=%%y
            )
        )

        REM Remover "%" do valor de CPU para obter somente o número
        set cpuusage=!cpuusage:%%=!

        REM Finalizar e remover o container
        docker stop tf_test_container
        docker rm tf_test_container

        REM Escrever resultado no CSV, incluindo uso de CPU e Memória
        if defined mean (
            echo /app/sw_model.tflite,%%m,%%c,!mean!,!stddev!,!ci!,!tput!,!cpuusage!,!memusage! >> %OUTPUT_CSV%
        ) else (
            echo Falha ao capturar o tempo de inferência para memória=%%m e CPU=%%c >> %OUTPUT_CSV%
        )
    )
)

echo Todos os testes foram concluídos. Resultados salvos em %OUTPUT_CSV%.
pause
