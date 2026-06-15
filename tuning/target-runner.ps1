$instanceName = $args[3]
$tunedParams = $args[4..($args.Length-1)]

python "$PSScriptRoot\..\irace.py" --instance $instanceName $tunedParams