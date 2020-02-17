import fs from "fs";
import parse from "csv-parse";


new Promise((resolve, reject) => {
    // Generate Random IDs
    let imageID = 0;
    let guesses = [null];
    fs.createReadStream(`./data/test.csv`)
        .pipe(parse({columns: true, comment: '#'}))
        .on('data', (data) => {
            imageID += 1;
            let guess = Math.floor(Math.random() * 10);
            guesses.push(guess);
        })
        .on('end', () => {
            // Write CSV Submission
            let writeStream = fs.createWriteStream('./submissions/random.csv');
            writeStream.write('ImageId,Label\n');
            for( let imageID=1; imageID<=guesses.length; imageID++ ) {
                writeStream.write(`${imageID},${guesses[imageID]}\n`);
            }
            console.log("wrote: ./submissions/random.csv");
            resolve(guesses)
        })
})
.then((guesses) => new Promise((resolve, reject) => {
    // Read Known Answer Data
    let answers = [null]; // ImageId starts at 0
    fs.createReadStream(`./data/answers.csv`)
        .pipe(parse({columns: true, comment: '#'}))
        .on('data', (data) => {
            answers.push( Number.parseInt(data['Label'], 10) );
        })
        .on('end', () => {
            resolve([guesses, answers]);
        })
    ;
}))
.then(([guesses, answers]) => {
    // Calculate Score
    let score = 0;
    for (let i = 1; i < guesses.length; i++) {
        if (guesses[i] === answers[i] && guesses[i] !== null) {
            score++;
        }
    }
    let accuracy = score / (guesses.length - 1);
    console.log(`Accuracy = ${score}/${guesses.length - 1} = ${(accuracy * 100).toFixed(2)}%`);
});