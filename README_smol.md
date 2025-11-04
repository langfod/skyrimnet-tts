
This is meant for secondary systems with older NVIDIA cards Maxwell and Pascal (GTX 1070 , 1660 etc).


Two things are needed:

1) CUDA 12.1.1 (yes this exact version)
  - https://developer.nvidia.com/cuda-12-1-1-download-archive
2) MS C++ compiler:


If you happen to have Visual Studio 2022 already installed, then run Visual Studio Installer and use Modify select the  "Desktop developement with C++".

other wise use:

`winget install --id=Microsoft.VisualStudio.2022.BuildTools -e --override "--passive --wait --add Microsoft.VisualStudio.Workload.VCTools;includeRecommended" --silent --accept-package-agreements --accept-source-agreements`




# CUIDA 12.1 Installer

If you happen to have other versions of CUDA installed then ad this to "Start.bat":
`set CUDA_PATH="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1"`


You only need these options:

![examples\cuda_12_1_installer.png](examples\cuda_12_1_installer.png)