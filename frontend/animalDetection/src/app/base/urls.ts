import { environment } from "../../environments/environment";
const api = environment.api;
export const URLs = {
    prediction: {
        uploadandprocess: api + 'upload_and_process',
        downloadprocessedimages: api + 'download_processed_images',
        generatereport: api + 'generate_report',
        deleteuserdata: api + 'delete_user_data'
    }
}
