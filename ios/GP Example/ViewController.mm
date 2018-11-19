//
//  ViewController.mm
//  IHGP Example
//
//  Created by Arno Solin on 02/05/2018.
//  Copyright © 2018 Arno Solin. All rights reserved.
//

#import <CoreMotion/CoreMotion.h>
#import <UIKit/UIKit.h>
#import "ViewController.h"
#include <Eigen/Dense>
#include <cmath>
#include <vector>
#include "Matern32model.hpp"
#include "InfiniteHorizonGP.hpp"

@interface ViewController ()

@end

@implementation ViewController
@synthesize label;
@synthesize figure1;
@synthesize figure2;
CMMotionManager *motionManager;
Matern32model model;
int counter;
std::vector< double > Y;
double px, py;
std::vector< double > Eft;
double Varft;
double mean;
std::vector< double > logLengthScales;
std::vector< double > logMagnSigma2s;

UIImage *plotbox;

- (void)viewDidLoad {
    [super viewDidLoad];
    // Do any additional setup after loading the view, typically from a nib.
    
    // Load plot box background
    plotbox = [UIImage imageNamed:@"seaborn"];
    
    // Set up model
    model.setMagnSigma2(1.0);
    model.setLengthScale(1.0);
    model.setSigma2(4.0);  // In the paper figures, \sigma_n^2 = 1 was used
    
    // Set up MotionManager
    motionManager = [[CMMotionManager alloc] init];
    
    // Sample number counter
    counter = 0;
    
    // Accelerometer data
    if (motionManager.accelerometerAvailable) {
        motionManager.accelerometerUpdateInterval = 0.01f;
        [motionManager startAccelerometerUpdatesToQueue:[NSOperationQueue mainQueue]
            withHandler:^(CMAccelerometerData *data, NSError *error) {
                                                
                // Add new sample
                if (!isnan(data.acceleration.x) & !isinf(data.acceleration.x)) {
                    Y.push_back(-9.81*data.acceleration.x);
                    counter++;
                }
                
                // Update
                if (counter > 200 && counter % 10 == 0) {
                    
                    // Do inference
                    InfiniteHorizonGP gp(0.01,model.getF(),model.getH(),model.getPinf(),model.getR(),model.getdF(),model.getdPinf(),model.getdR());
                    
                    // Calculate mean to keep the figure in-view
                    mean = 0;
                    for (int k = counter-200; k <= counter; k++) {
                        mean += Y[k];
                    }
                    mean /= 200;
                    
                    // Loop through data
                    for (int k = counter-200; k <= counter; k++) {
                        gp.update(Y[k]-mean);
                    }
                    
                    // Pull out the gradient (account for log-transformation)
                    double logMagnSigma2 = log(model.getMagnSigma2());
                    double logLengthScale = log(model.getLengthScale());
                    double dLikdlogMagnSigma2 = model.getMagnSigma2() * gp.getLikDeriv()(1);
                    double dLikdlogLengthScale = model.getLengthScale() * gp.getLikDeriv()(2);
                    
                    // Do the gradient descent step
                    logMagnSigma2 = logMagnSigma2 - 0.1*dLikdlogMagnSigma2;
                    logLengthScale = logLengthScale - 0.01*dLikdlogLengthScale;
                    
                    // Introduce contraints to keep the behavior better in control
                    if (logMagnSigma2 < -6) { logMagnSigma2 = -6; } else if (logMagnSigma2 > 4) { logMagnSigma2 = 4; }
                    if (logLengthScale < -2) { logLengthScale = -2; } else if (logLengthScale > 2) { logLengthScale = 2; }

                    // Update the model
                    model.setMagnSigma2(exp(logMagnSigma2));
                    model.setLengthScale(exp(logLengthScale));
                    
                    // Check if this went bad and re-initialize
                    if (isnan(model.getMagnSigma2()) | isnan(model.getLengthScale()) |
                        isinf(model.getMagnSigma2()) | isinf(model.getLengthScale())) {
                        model.setMagnSigma2(1.0);
                        model.setLengthScale(1.0);
                        NSLog(@"Bad parameters.");
                    }

                    // Push previous hyperparameters to history
                    logMagnSigma2s.push_back(logMagnSigma2);
                    logLengthScales.push_back(logLengthScale);
                    
                    // Pull out the marginal mean and variance estimates
                    Eft = gp.getEft();
                    Varft = gp.getVarft();
                    
                    // Report in its own thread
                    [[NSOperationQueue mainQueue] addOperationWithBlock:^{
                        
                        // Update labels
                        self->label.text = [NSString stringWithFormat:@"log magnitude: \n %+.02f\n\nlog length-scale: \n %+.02f\n\nSamples seen: \n%i",log(model.getMagnSigma2()),log(model.getLengthScale()),counter];

                        // Update plot of hyperparameters
                        UIImage* img = [self imageByDrawingHyperparametersOnImage:plotbox];
                        [self.figure2 setImage: img];
                        
                    }];
                    
                }

                // Draw on every 10th observation
                if (counter % 10 == 0) {
                    // Report in its own thread
                    [[NSOperationQueue mainQueue] addOperationWithBlock:^{
                        // Update plot of data and estimate
                        if (!isnan(model.getMagnSigma2()) & !isnan(model.getLengthScale()) &
                            !isinf(model.getMagnSigma2()) & !isinf(model.getLengthScale())) {
                            [self.figure1 setImage: [self imageByDrawingObservations]];
                        }
                     }];
                }
            }];
    }
}


- (void)didReceiveMemoryWarning {
    [super didReceiveMemoryWarning];
    // Dispose of any resources that can be recreated.
}

- (UIImage *)imageByDrawingHyperparametersOnImage:(UIImage *)image
{
    // begin a graphics context of sufficient size
    UIGraphicsBeginImageContext(image.size);
    
    // draw original image into the context
    [image drawAtPoint:CGPointZero];
    
    // get the context for CoreGraphics
    CGContextRef ctx = UIGraphicsGetCurrentContext();
    
    // set stroking color and draw circle
    //[[UIColor redColor] setStroke];
    CGContextSetRGBStrokeColor(ctx, (42.0/255.0), (171.0/255.0), (97.0/255.0), 1);
    CGContextSetRGBFillColor(ctx, (42.0/255.0), (171.0/255.0), (97.0/255.0), 1);
    
    // The plot area
    const double PLOT_W = 667;
    const double PLOT_H = 650;
    const double CORNER_X = 76;
    const double CORNER_Y = 30;
    
    double px = 0;
    double py = 0;
    
    // Set color
    CGContextSetRGBFillColor(ctx, (42.0/255.0), (171.0/255.0), (97.0/255.0), 1);

    // Plot trail
    for (unsigned long j=logLengthScales.size()-1; j>logLengthScales.size()-10; j--) {
        
        // The current point
        px = logLengthScales[j];
        py = logMagnSigma2s[j];
        
        // Introduce contraints to keep the dot on the plot
        if (py < -6) { py = -6; } else if (py > 4) { py = 4; }
        if (px < -2) { px = -2; } else if (px > 2) { px = 2; }
        
        // Keep inside window, where xlim([-2 2]), ylim([-6 4])
        px = CORNER_X + (px+2.0)/4.0*PLOT_W - 20;
        py = CORNER_Y + PLOT_H - (py+6.0)/10.0*PLOT_H - 20;
        
        // Make circle
        CGRect circleRect = CGRectMake(px, py, 40, 40);
        circleRect = CGRectInset(circleRect, 10, 10);
        
        // Draw circle
        CGContextFillEllipseInRect(ctx, circleRect);
        
        // Set color
        CGContextSetRGBFillColor(ctx, (42.0/255.0), (171.0/255.0), (97.0/255.0), .25);
        
        if (j==0) { break; };
    
    }
    
    // make image out of bitmap context
    UIImage *retImage = UIGraphicsGetImageFromCurrentImageContext();
    
    // free the context
    UIGraphicsEndImageContext();
    
    return retImage;
}


- (UIImage *)imageByDrawingObservations
{
    // Make empty canvas
    UIGraphicsBeginImageContextWithOptions(CGSizeMake(200, 200), NO, 0.0);
    UIImage *image = UIGraphicsGetImageFromCurrentImageContext();
    UIGraphicsEndImageContext();
    
    // begin a graphics context of sufficient size
    UIGraphicsBeginImageContext(image.size);
    
    // draw original image into the context
    [image drawAtPoint:CGPointZero];
    
    // get the context for CoreGraphics
    CGContextRef ctx = UIGraphicsGetCurrentContext();
    
    if (Y.size() > 200) {

        // Marginal standard deviation
        double stdft = sqrt(Varft);
        
        // Draw the 95% quantiles
        double px = 0.0;
        double py = (Eft[0]+1.96*stdft+10.0)/20.0*image.size.height;
        CGContextMoveToPoint(UIGraphicsGetCurrentContext(), px,py);
        for (int k=1; k<Eft.size(); k++) {
            px = k/200.0*image.size.width;
            py = (Eft[k]+1.96*stdft+10.0)/20.0*image.size.height;
            CGContextAddLineToPoint(ctx, px, py);
        }
        for (int k=(int)Eft.size()-1; k>=0; k--) {
            px = k/200.0*image.size.width;
            py = (Eft[k]-1.96*stdft+10.0)/20.0*image.size.height;
            CGContextAddLineToPoint(ctx, px, py);
        }
        py = (Eft[0]+1.96*stdft+10.0)/20.0*image.size.height;
        CGContextAddLineToPoint(ctx, 0.0, py);
        CGContextSetRGBFillColor(ctx, (68.0/255.0), (114.0/255.0), (181.0/255.0), 0.33);
        CGContextFillPath(ctx);
        
    }
        
    // Cursor position
    unsigned long cursor = 0;
    if (Y.size()>200) {
        cursor = Y.size()-200;
    }
    
    // Set stroking color for observations
    //[[UIColor redColor] setStroke];
    CGContextSetRGBStrokeColor(ctx, (211.0/255.0), (67.0/255.0), (78.0/255.0), 1);

    // Loop through every data point in the window
    for (int k=0; k<200; k++) {
    
        // Keep inside window
        px = (k)/200.0*image.size.width - 2.5;
        py = (Y[cursor+k]-mean+10)/20.0*image.size.height - 2.5;
        
        // Draw circle
        CGRect circleRect = CGRectMake(px, py, 5, 5);
        circleRect = CGRectInset(circleRect, 2, 2);
        CGContextStrokeEllipseInRect(ctx, circleRect);
        
        if (k==Y.size()-1) { break; }
        
    }
    
    if (Y.size() > 200) {

        // Set line properties
        CGContextSetRGBStrokeColor(ctx, (68.0/255.0), (114.0/255.0), (181.0/255.0), 1);
        CGContextSetLineWidth(ctx, 1);
        
        // Draw Eft
        px = 0.0;
        py = (Eft[0]+10.0)/20.0*image.size.height;
        CGContextMoveToPoint(ctx, px, py);
        
        // Loop through every data point in Eft
        for (int k=1; k<Eft.size(); k++) {
            px = k/200.0*image.size.width;
            py = (Eft[k]+10.0)/20.0*image.size.height;
            CGContextAddLineToPoint(UIGraphicsGetCurrentContext(), px, py);
        }
        CGContextStrokePath(UIGraphicsGetCurrentContext());
        
    }
    
    // make image out of bitmap context
    UIImage *retImage = UIGraphicsGetImageFromCurrentImageContext();
    
    // free the context
    UIGraphicsEndImageContext();
    
    return retImage;
}



@end
