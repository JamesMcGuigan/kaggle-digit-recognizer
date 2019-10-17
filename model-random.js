import fs from "fs";
import parse from "csv-parse";

let writeStream = fs.createWriteStream('./submissions/random.csv');
writeStream.write('ImageId,Label\n');

let imageID = 0;
fs.createReadStream(`./data/test.csv`)
    .pipe(parse({ columns: true }))
    .on('data', (data) => {
        imageID += 1;
        let guess = Math.floor(Math.random() * 10);
        writeStream.write(`${imageID},${guess}\n`);
    })
    .on('end', () => {
        console.info("END");
    });