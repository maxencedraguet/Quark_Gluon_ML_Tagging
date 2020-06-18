file(REMOVE_RECURSE
  "../x86_64-centos7-gcc8-opt/include/MyAnalysis"
  "../x86_64-centos7-gcc8-opt/bin/ATestRun_eljob.py"
  "../x86_64-centos7-gcc8-opt/include/MyAnalysis"
)

# Per-language clean rules from dependency scanning.
foreach(lang )
  include(CMakeFiles/MyAnalysisHeaderInstall.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
