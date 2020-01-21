import { Component } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import {FormGroup } from '@angular/forms';


@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent {
  title = 'app';
  app: FormGroup;
  prediction:any;
  result:any;
  constructor(private httpClient: HttpClient) {
  }

  ngOnInit() {
  }


  onSubmit(formData) {
    console.log(formData);
    this.httpClient.post('http://<host-ip>:5002/predict',
    formData).subscribe(data => {
      this.prediction = data as JSON;
      this.result=this.prediction.Prediction;
      console.log(this.prediction.Prediction);
    })
  }

}
