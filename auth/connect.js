console.log('Node is installed!');
// Imports the Google Cloud client library
const {Storage} = require('@google-cloud/storage');

// Your Google Cloud Platform project ID
const projectId = 'song-transcriber';

// Creates a client
const storage = new Storage({
  projectId: projectId,
});

const bucketName = 'song-repo';
const srcFilename = 'fma_small/000/000002.mp3';
const destFilename = 'file.mp3'; // upload to the same folder?

const options = {
  // The path to which the file should be downloaded, e.g. "./file.txt"
  destination: destFilename,
};

// Downloads the file

async function doSomething() {
await storage
  .bucket(bucketName)
  .file(srcFilename)
  .download(options);

  console.log(
  `gs://${bucketName}/${srcFilename} downloaded to ${destFilename}.`
);
}

doSomething()
.then(console.log)
  .catch(console.error)

