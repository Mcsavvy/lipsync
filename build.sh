# rm -rf build/*.py build/gui build/models
pyarmor gen \
    --platform windows.x86_64 \
    --platform linux.x86_64 \
    --platform darwin.x86_64 \
    --output build \
    --recursive \
    *.py gui models
cp config.ini build
zip -r build.zip build/gui build/models build/pyarmor_runtime_* build/config.ini build/*.py