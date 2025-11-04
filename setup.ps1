$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")

cmd.exe /c "call `"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat`" && set > %temp%\vcvars.txt"


Get-Content "$env:temp\vcvars.txt" | Foreach-Object {
  if ($_ -match "^(.*?)=(.*)$") {
    Set-Content "env:\$($matches[1])" $matches[2]
  }
}