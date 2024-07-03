from invoke import main

for i in main(
    face="taylor.mp4", audio="cardi.mp3", output="cardi_taylor.mp4", quality="Enhanced"
):
    print(i)