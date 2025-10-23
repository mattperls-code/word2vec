const fs = require("fs")
const path = require("path")

const cleanCorpus = (inputPath, outputPath) => {
    const rawText = fs.readFileSync(inputPath, "utf-8")

    let text = rawText.toLowerCase()

    text = text.replace(/’/g, "'")

    text = text.replace(/\b([a-z]+)'s\b/g, "$1")

    text = text.replace(/(^|[^a-z])'|'(?![a-z])/g, " ")

    text = text.replace(/[^a-z0-9'\s]+/g, " ")

    text = text.replace(/\s+/g, " ").trim()

    fs.writeFileSync(outputPath, text, "utf-8")

    console.log(`Cleaned corpus written to ${outputPath}`)
}

const inputPath = path.join(__dirname, "rawCorpus/fairy_tales.txt")
const outputPath = path.join(__dirname, "cleanCorpus/fairy_tales.txt")

cleanCorpus(inputPath, outputPath)
