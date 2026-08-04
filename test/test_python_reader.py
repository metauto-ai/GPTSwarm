from swarm.environment.tools.reader.readers import FileReader, GeneralReader


def test_python_reader_returns_source_without_executing(tmp_path):
    """Reading a .py attachment must return its source text, not run it."""
    marker = tmp_path / "marker"
    attachment = tmp_path / "attachment.py"
    attachment.write_text(
        f"open({str(marker)!r}, 'w').write('ran')\nx = 1 + 1\n"
    )

    # PythonReader.parse returns the source as plain text.
    source = FileReader().read_file(str(attachment))
    assert isinstance(source, str)
    assert "x = 1 + 1" in source
    # The attachment must not have been executed.
    assert not marker.exists()

    # GeneralReader.read must treat .py like any other text file (no 3-tuple indexing).
    rendered = GeneralReader().read("describe the file", str(attachment))
    assert isinstance(rendered, str)
    assert "x = 1 + 1" in rendered
    assert not marker.exists()
