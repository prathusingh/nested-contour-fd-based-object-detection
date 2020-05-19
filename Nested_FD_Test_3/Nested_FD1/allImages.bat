for %%f in (stop*.jpg) do (

            echo %%f
start Nested_FD_Test_3_FinalOutModel.exe %%f 5 1 1 1024 256 50 30 10 100

)

for %%f in (w*.jpg) do (

            echo %%f
start Nested_FD_Test_3_FinalOutModel.exe %%f 5 1 1 1024 256 50 30 10 10

)

for %%f in (y*.jpg) do (

            echo %%f
start Nested_FD_Test_3_FinalOutModel.exe %%f 5 1 2 1024 256 50 30 10 10


)