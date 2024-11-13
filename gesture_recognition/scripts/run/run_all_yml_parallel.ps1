
# Caminho para a pasta que contém os arquivos .yml
$ymlFolder = "configs\tests_experiment"

# Caminho para o script Python
$pythonScript = ".\src\pipeline.py"

$workingDirectory = "K:\Master\repositories\SignWritingAI\sw-modeling"

# Obter todos os arquivos .yml na pasta
$ymlFiles = Get-ChildItem -Path $ymlFolder -Filter *.yml

# Definir o número máximo de tarefas paralelas
$maxParallelJobs = 3

# Iterar sobre cada arquivo .yml
foreach ($ymlFile in $ymlFiles) {
    # Verifica quantos jobs estão ativos e espera até que haja menos de $maxParallelJobs
    while ((Get-Job | Where-Object { $_.State -eq 'Running' }).Count -ge $maxParallelJobs) {
        Start-Sleep -Seconds 1
    }

    # Caminho completo do arquivo yml
    $ymlPath = $ymlFile.FullName

    # Iniciar o trabalho em paralelo
    $job = Start-Job -ScriptBlock {
        param ($pythonScript, $ymlPath, $workingDirectory)
        Set-Location $workingDirectory
        python $pythonScript $ymlPath
    } -ArgumentList $pythonScript, $ymlPath, $workingDirectory

    # Exibir o progresso
    Write-Host "Iniciado: $($ymlFile.Name)"
}

# Aguardar a conclusão de todos os trabalhos antes de sair
Get-Job | Wait-Job

# Exibir as saídas de todos os trabalhos
foreach ($job in Get-Job) {
    $output = Receive-Job -Job $job
    Write-Host "Resultado do trabalho $($job.Id):"
    Write-Host $output
    Remove-Job -Job $job
}