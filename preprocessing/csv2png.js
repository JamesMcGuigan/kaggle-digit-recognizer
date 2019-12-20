import _ from 'lodash';
import fs from 'fs';
import Jimp from 'jimp';
import parse from "csv-parse";
import Promise from 'bluebird';
import countLinesInFile from 'count-lines-in-file';

/**
 * @param   {number[]} data
 * @param   {object}   size
 * @param   {string}   filename
 * @returns {Promise}  image
 */
export function grayscale2image(data, size, filename=null) {
    return new Promise((resolve, reject) => {
        data = _(data)
            .omit(['label'])
            .values()
            .map(Number)
            .chunk(size.w)
            .value()
        ;
        new Jimp(size.w, size.h, 255, (error, image) => {
            image
                .scan(0, 0, image.bitmap.width, image.bitmap.height, function (x, y, idx) {
                    let grayscale = data[y][x];
                    if( size.invert ) { grayscale = 255 - grayscale; }

                    ['red', 'green', 'blue'].forEach((color, offset) => {
                        this.bitmap.data[idx + offset] = grayscale;
                    });
                    this.bitmap.data[idx + 4] = 255;  // alpha
                })
                .greyscale()  // set greyscale
            ;

            if( filename ) {
                image.writeAsync(filename)
                    .then((image) => {
                        console.info(filename, image.hash(2));
                        resolve(image)
                    })
                    .catch((error) => reject(error))
                ;
            }
        });
    })
}


const size = {
    w: 28,
    h: 28,
    invert: false
};
// ['test'].forEach((testtrain) => {  // time = 60min
['test', 'train'].forEach((testtrain) => {
    // BUGFIX: ImageDataBunch sorts filenames alphabetically, not numerically, causing image/id mismatch
    countLinesInFile(`./data/${testtrain}.csv`, (error, lineCount) => {
        const lineCountDigits = Math.ceil(Math.log10( lineCount+1));
        const padID = (imageID) => String(imageID).padStart(lineCountDigits, '0');

        let imageID = 0;
        fs.createReadStream(`./data/${testtrain}.csv`)
            .pipe(parse({
                columns: true,
            }))
            .on('data', (data) => {
                imageID += 1;
                let filename = `./data/images/${testtrain}/${data.label || ''}/${padID(imageID)}.png`;   // Imagenet format
                filename     = filename.replace(/\/+/g, "/");                                            // strip double // in path
                fs.exists(filename, (exists) => {
                    console.log( (exists ? "skip:  " : "write: ") + filename );
                    if( !exists ) { grayscale2image(data, size, filename); }
                });
            })
            .on('end', () => {
                console.info("END");
            });
    });
});