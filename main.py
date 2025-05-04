import typer

app = typer.Typer()

@app.command()
def main():
    print("Hello, World!")

