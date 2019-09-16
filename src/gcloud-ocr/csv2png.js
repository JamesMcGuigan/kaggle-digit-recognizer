import parse from "csv-parse";
import jetpack from "fs-jetpack";
import Promise from 'bluebird';
import fs from 'fs';
import _ from 'lodash';
import jimp from 'jimp';

/**
 *
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
        new jimp(size.w, size.h, 255, (error, image) => {
            image
                .scan(0, 0, image.bitmap.width, image.bitmap.height, function (x, y, idx) {
                    let grayscale = data[y][x];
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
};
// ['test'].forEach((testtrain) => {  // time = 60min
['test', 'train'].forEach((testtrain) => {
    let imageID = 0;
    fs.createReadStream(`./data/${testtrain}.csv`)
        .pipe(parse({
            columns: true,
        }))
        .on('data', (data) => {
            imageID += 1;
            let filename = `./data-images/${testtrain}/${imageID}.png`;
            fs.exists(filename, (exists) => {
                if( !exists ) { grayscale2image(data, size, filename); }  
            });
        })
        .on('end', () => {
            console.info("END");
        });
});