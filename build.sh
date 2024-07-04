pyarmor gen \
    --platform windows.x86_64 \
    --platform linux.x86_64 \
    --platform darwin.x86_64 \
    --restrict \
    --assert-import \
    --output build \
    --recursive \
    *.py gui models
zip -r build.zip build