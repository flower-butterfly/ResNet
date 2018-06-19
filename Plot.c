void Plot(){
    const char *name[2]={"plain_CNN.txt","ResNet_CNN.txt"};
    ifstream infile1(name[0]);
    ifstream infile2(name[1]);

    Color_t col[2]={kRed,kBlue};
const int NN=10010;
int N=0;
double plain[6][NN];
double ResNet[6][NN];

//  plain network,    and the resnet network
//  index  loss learningrate  runtime  trainerror   testerror 
//


while(infile2){
    for(int j=0;j<6;j++){
        infile1>>plain[j][N];    
        infile2>>ResNet[j][N];
    }
    N++;
}
N--;

cout<<"The Value of N is:  "<<N<<endl;

TCanvas *myc1=new TCanvas("myc1","myc1",800,600);
TGraph *gr1=new TGraph(N,plain[0],plain[1]);
TGraph *gr2=new TGraph(N,ResNet[0],ResNet[1]);

TMultiGraph *mgr=new TMultiGraph();
mgr->Add(gr1);
mgr->Add(gr2);

gr1->SetLineColor(col[0]);
gr1->SetMarkerColor(col[0]);
gr2->SetLineColor(col[1]);
gr2->SetMarkerColor(col[1]);


mgr->Draw("APL");
mgr->GetXaxis()->SetTitle("iteration");
mgr->GetYaxis()->SetTitle("loss");
mgr->GetXaxis()->SetTitleSize(0.04);
mgr->GetYaxis()->SetTitleSize(0.04);
mgr->GetYaxis()->SetRangeUser(0,20);

mgr->Draw("APL");
myc1->Update();

TLegend *lg=new TLegend(0.4,0.6,0.9,0.9);
lg->AddEntry(gr1,"Plain Covolutional NetWork","lp");
lg->AddEntry(gr2,"ResNet NetWork","lp");
lg->SetFillStyle(0);
lg->Draw("same");







TCanvas *myc2=new TCanvas("myc2","myc2",800,600);
TGraph *_gr1=new TGraph(N,plain[0],plain[4]);
TGraph *_gr2=new TGraph(N,ResNet[0],ResNet[4]);

TMultiGraph *_mgr=new TMultiGraph();
_mgr->Add(_gr1);
_mgr->Add(_gr2);

_gr1->SetLineColor(col[0]);
_gr1->SetMarkerColor(col[0]);
_gr2->SetLineColor(col[1]);
_gr2->SetMarkerColor(col[1]);


_mgr->Draw("APL");
_mgr->GetXaxis()->SetTitle("iteration");
_mgr->GetYaxis()->SetTitle("train error");
_mgr->GetXaxis()->SetTitleSize(0.04);
_mgr->GetYaxis()->SetTitleSize(0.04);
_mgr->GetYaxis()->SetRangeUser(0,100);

_mgr->Draw("APL");
myc2->Update();

TLegend *_lg=new TLegend(0.4,0.6,0.9,0.9);
_lg->AddEntry(_gr1,"Plain Covolutional NetWork","lp");
_lg->AddEntry(_gr2,"ResNet NetWork","lp");
_lg->SetFillStyle(0);
_lg->Draw("same");




TCanvas *myc3=new TCanvas("myc3","myc3",800,600);
TGraph *m_gr1=new TGraph(N,plain[0],plain[5]);
TGraph *m_gr2=new TGraph(N,ResNet[0],ResNet[5]);

TMultiGraph *m_mgr=new TMultiGraph();
m_mgr->Add(m_gr1);
m_mgr->Add(m_gr2);

m_gr1->SetLineColor(col[0]);
m_gr1->SetMarkerColor(col[0]);
m_gr2->SetLineColor(col[1]);
m_gr2->SetMarkerColor(col[1]);


m_mgr->Draw("APL");
m_mgr->GetXaxis()->SetTitle("iteration");
m_mgr->GetYaxis()->SetTitle("test error");
m_mgr->GetXaxis()->SetTitleSize(0.04);
m_mgr->GetYaxis()->SetTitleSize(0.04);
m_mgr->GetYaxis()->SetRangeUser(0,100);

m_mgr->Draw("APL");
myc2->Update();

TLegend *m_lg=new TLegend(0.4,0.6,0.9,0.9);
m_lg->AddEntry(m_gr1,"Plain Covolutional NetWork","lp");
m_lg->AddEntry(m_gr2,"ResNet NetWork","lp");
m_lg->SetFillStyle(0);
m_lg->Draw("same");



myc1->Print("Plot/mnist_loss.png");
myc2->Print("Plot/mnist_train_error.png");
myc3->Print("Plot/mnist_test_error.png");

}

















